# Copied from imitation-learning-dp

import time

import gym
import torch
from gym import spaces
import numpy as np
import spatialmath as sm
import cv2
import pickle
import mujoco
import mujoco.viewer
import glfw

from imitation_learning_dp.arm.robot import Robot, UR5e
from imitation_learning_dp.arm.motion_planning import LinePositionParameter, OneAttitudeParameter, CartesianParameter, \
    QuinticVelocityParameter, TrajectoryParameter, TrajectoryPlanner
from imitation_learning_dp.utils import mj


class UR5GrabImageEnv(gym.Env):
    metadata = {"render.mode": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self,
                 render_size=96,
                 ):
        super().__init__()
        self.sim_hz = 500
        self.control_hz = 25
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=1,
                shape=(3, render_size, render_size),
                dtype=np.float32
            ),
            'agent_pos': spaces.Box(
                low=np.array([1.2, 0.1], dtype=np.float32),
                high=np.array([1.6, 1.1], dtype=np.float32),
                shape=(2,),
                dtype=np.float32
            )
        })
        self.action_space = spaces.Box(
            low=np.array([1.2, 0.1], dtype=np.float32),
            high=np.array([1.6, 1.1], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )
        self.latest_action = None
        self.render_cache = None

        self.mj_model: mujoco.MjModel = None
        self.mj_data: mujoco.MjData = None
        self.robot: Robot = None
        self.ur5e_joint_names = []
        self.robot_q = np.zeros(6)
        self.robot_T = sm.SE3()
        self.T0 = sm.SE3()
        self.obj_t = np.zeros(3)

        self.step_num = 0
        self.mj_renderer: mujoco.Renderer = None
        self.mj_viewer: mujoco.viewer.Handle = None

    def reset(self):
        self.mj_model = mujoco.MjModel.from_xml_path("./imitation_learning_dp/assets/scenes/scene.xml")
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.robot = UR5e()
        self.robot.set_base(mj.get_body_pose(self.mj_model, self.mj_data, "ur5e_base").t)
        Te = sm.SE3(1.25, 0.6, 0.8) * sm.SE3.Rz(np.pi / 2) * sm.SE3.Rx(np.pi / 2)
        self.robot_q = self.robot.ikine(Te)
        self.ur5e_joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint",
                                 "wrist_2_joint", "wrist_3_joint"]
        [mj.set_joint_q(self.mj_model, self.mj_data, jn, self.robot_q[i]) for i, jn in enumerate(self.ur5e_joint_names)]
        mujoco.mj_forward(self.mj_model, self.mj_data)
        mj.attach(self.mj_model, self.mj_data, "attach", "2f85", self.robot.fkine(self.robot_q))
        self.robot.set_tool(np.array([0.0, 0.0, 0.15]))
        Te = sm.SE3(1.4, 0.6, 0.8) * sm.SE3.Rz(np.pi / 2) * sm.SE3.Rx(np.pi / 2)
        self.robot_q[:] = self.robot.ikine(Te)
        self.robot.set_joint(self.robot_q)
        self.robot_T = self.robot.fkine(self.robot_q)
        self.T0 = self.robot_T.copy()

        px = np.random.uniform(low=1.4, high=1.55)
        py = np.random.uniform(low=0.2, high=1.0)
        while 0.5 < py < 0.7:
            py = np.random.uniform(low=0.2, high=1.0)
        T_Box = sm.SE3.Trans(px, py, 0.8)
        mj.set_free_joint_pose(self.mj_model, self.mj_data, "Box", T_Box)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.obj_t = mj.get_body_pose(self.mj_model, self.mj_data, "Box").t

        self.mj_renderer = mujoco.Renderer(self.mj_model, height=96, width=96)
        self.mj_renderer.update_scene(self.mj_data, 0)
        self.mj_viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)

        self.step_num = 0

        observation = self._get_obs()
        return observation

    def step(self, action):
        dt = 1.0 / self.sim_hz
        n_steps = self.sim_hz // self.control_hz
        if action is not None:
            self.latest_action = action
            for i in range(n_steps):
                Ti = sm.SE3.Trans(action[0], action[1], 0.8) * sm.SE3(sm.SO3(self.T0.R))
                self.robot.move_cartesian(Ti)
                joint_position = self.robot.get_joint()
                self.mj_data.ctrl[:6] = joint_position
                mujoco.mj_step(self.mj_model, self.mj_data)

        observation = self._get_obs()
        self.obj_t = mj.get_body_pose(self.mj_model, self.mj_data, "Box").t
        distance = np.linalg.norm(self.robot_T.t[:2] - self.obj_t[:2])
        reward = 1 / (distance + 1)
        done = False
        if distance < 0.01:
            done = True
        info = self._get_info()

        self.step_num += 1
        if self.step_num > 4000:
            done = True

        return observation, reward, done, info

    def render(self, mode):
        assert mode == 'rgb_array'

        if self.render_cache is None:
            self._get_obs()
        return self.render_cache

    def close(self):
        if self.mj_viewer is not None:
            self.mj_viewer.close()
        super().close()

    def seed(self, seed=None):
        return super().seed(seed)

    def run(self):
        time0 = 0.001
        T0 = self.robot.get_cartesian()
        t0 = T0.t
        R0 = sm.SO3(T0.R)
        t1 = t0.copy()
        R1 = R0.copy()
        planner0 = self.cal_planner(t0, R0, t1, R1, time0)

        time1 = 1.0
        t2 = t1.copy()
        t2[0] = self.obj_t[0] - 0.1
        R2 = R1.copy()
        planner1 = self.cal_planner(t1, R1, t2, R2, time1)

        time2 = 2.0
        t3 = self.obj_t + np.array([-0.1, 0.0, 0.0])
        R3 = R2.copy()
        planner2 = self.cal_planner(t2, R2, t3, R3, time2)

        time3 = 1.0
        t4 = self.obj_t.copy()
        R4 = R3.copy()
        planner3 = self.cal_planner(t3, R3, t4, R4, time3)

        time4 = 1.0
        t5 = t4.copy()
        R5 = R4.copy()
        planner4 = self.cal_planner(t4, R4, t5, R5, time4)

        time_array = np.array([0, time0, time1, time2, time3, time4])
        planner_array = [planner0, planner1, planner2, planner3, planner4]
        total_time = np.sum(time_array)

        time_step_num = round(total_time / self.mj_model.opt.timestep) + 1
        every_step_num = 20
        every_epoch_num = time_step_num // every_step_num
        times = np.linspace(0, total_time, time_step_num)
        desired_poses = np.zeros((time_step_num, self.robot.dof))

        imgs = np.zeros((every_epoch_num, 96, 96, 3), dtype=np.uint8)
        states = np.zeros((every_epoch_num, 2))
        actions = np.zeros((every_epoch_num, 2))

        time_cumsum = np.cumsum(time_array)
        joint_position = self.robot_q.copy()
        for i, timei in enumerate(times):
            for j in range(len(time_cumsum)):
                if timei < time_cumsum[j]:
                    planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                    self.robot.move_cartesian(planner_interpolate)
                    joint_position = self.robot.get_joint()
                    break

            desired_poses[i, :] = joint_position

        data_num = 0
        time_num = 0
        while self.mj_viewer.is_running():
            step_start = time.time()

            self.mj_renderer.update_scene(self.mj_data, 0)
            image = self.mj_renderer.render()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            window_name = 'Animation'
            cv2.imshow(window_name, image)
            cv2.waitKey(1)

            if time_num % every_step_num == 0:
                self.mj_renderer.update_scene(self.mj_data, 0)

                joint_state = np.array(
                    [mj.get_joint_q(self.mj_model, self.mj_data, name)[0] for name in self.ur5e_joint_names])
                state = self.robot.fkine(joint_state).t[:2]
                action = self.robot.fkine(desired_poses[time_num, :]).t[:2]

                imgs[data_num, ...] = self.mj_renderer.render()
                states[data_num, ...] = state
                actions[data_num, ...] = action
                data_num += 1

            time_num += 1
            if time_num >= time_step_num - every_step_num:
                break

            self.mj_data.ctrl[:6] = desired_poses[time_num, :]
            mujoco.mj_step(self.mj_model, self.mj_data)

            self.mj_viewer.sync()
            time_until_next_step = self.mj_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        self.mj_viewer.close()
        self.mj_renderer.close()
        return {
            'imgs': imgs,
            'states': states,
            'actions': actions
        }

    def cal_planner(self, t0, R0, t1, R1, time):
        position_parameter = LinePositionParameter(t0, t1)
        attitude_parameter = OneAttitudeParameter(R0, R1)
        cartesian_parameter = CartesianParameter(position_parameter, attitude_parameter)
        velocity_parameter = QuinticVelocityParameter(time)
        trajectory_parameter = TrajectoryParameter(cartesian_parameter, velocity_parameter)
        trajectory_planner = TrajectoryPlanner(trajectory_parameter)
        return trajectory_planner

    def _get_obs(self):
        self.mj_renderer.update_scene(self.mj_data, 0)
        img = self.mj_renderer.render()

        for i in range(len(self.ur5e_joint_names)):
            self.robot_q[i] = mj.get_joint_q(self.mj_model, self.mj_data, self.ur5e_joint_names[i])
        self.robot_T = self.robot.fkine(self.robot_q)
        agent_pos = self.robot_T.t[:2]
        img_obs = np.moveaxis(img.astype(np.float32) / 255, -1, 0)
        obs = {
            'image': img_obs,
            'agent_pos': agent_pos
        }
        self.render_cache = img
        return obs

    def _get_info(self):
        info = {
            'pos_agent': self.robot_T.t[:2],
            'goal_pose': self.obj_t
        }
        return info
