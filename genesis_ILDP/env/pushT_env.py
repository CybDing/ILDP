import time
import os
import sys
from typing import Dict, List, Optional, Tuple, Union

import gym
import torch
from gym import spaces
import numpy as np

import cv2
import pickle
import genesis as gs
import glfw
import requests
import json
from copy import deepcopy

from genesis_ILDP.utils.cuda import *
from genesis_ILDP.config.env_config import *
from shapely.geometry import Polygon

class PushTEnv(gym.Env):
    # TODO close(), 
    metadata = {"render.mode": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self,
                 sim_hz = 100, 
                 control_hz = 10, 
                 render_size=(96, 96),
                 xlim=.2,
                 ylim=.2,
                 seed=None,
                 model_path=env_path,
                 fps = 25,
                 show_fps = True,
                 device = None,
                 done_ratio = 0.7,
                 spawn_center=(-0.3, 0.3),  # Center of spawn region (x, y)
                 spawn_range_scale=0.6,      # Scale factor for spawn range (1.0 = use xlim/ylim)
                 ):

        self.render_size = render_size
        self.sim_hz = float(sim_hz) # sim_hz represent the actual simulation timestep for robotic manipulation 
        self.control_hz = float(control_hz)# control_hz represent the control frequency of receiving a new action from controller
        self.steps_per_render = int(self.sim_hz) // fps # control how many frames should be recording
        print("step_per_render: ", self.step_per_render) # render after how many global steps(refresh when a new step() is called for simplification)



        self.is_init = False
        self._seed = seed
        self.scene = None
        self.n_envs = None
        self.np_random = None
        self.block_lim = {'xlim': xlim, 'ylim': ylim}
        self.path = model_path
        self.env_seed = None
        self.fps = fps
        self.show_fps = show_fps
        self.device = None
        self.done_ratio = done_ratio
        self.spawn_center = spawn_center
        self.spawn_range_scale = spawn_range_scale
        self.home_pos = None

        self.ini_delta_dis: Union[float, np.ndarray] = 0.0
        self.ini_delta_ang: Union[float, np.ndarray] = 0.0 
        self.reward0: Optional[float] = None
        self.poses: Optional[Dict[str, torch.Tensor]] = None  
        self.keypoints: Optional[Dict[str, np.ndarray]] = None
        
        self.observation_space = spaces.Dict({
            'images': spaces.Box(
                low=0.,
                high=1.,
                shape=(3,) + render_size,
                dtype = np.float64
            ),
            'agent_pos': spaces.Box(
                low=np.array([0.1, 0.1], dtype=np.float32),
                high=np.array([xlim+0.1, ylim+0.1], dtype=np.float32),
                shape=(2, ),
                dtype=np.float32
            )
        })
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([xlim+0.2, ylim+0.2], dtype=np.float32),
            shape=(2, ),
            dtype=np.float32
        )

        self.seed()


    def start(self, n_envs=1, show_interact_viewer=False, show_camera=False, 
              seed=None, env_separate=False):
        assert self.is_init == False 
        self.n_envs = n_envs
        
        gs.init(
            seed = self.env_seed if self.env_seed is not None else 0,
            backend = gs.gpu, 
            performance_mode = True
        )
        self.scene = gs.Scene(
            show_FPS=False,
            sim_options=gs.options.SimOptions(dt=1./self.sim_hz, substeps=1),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=30,
                camera_pos=(2, 2, 1.5),
                camera_lookat=(0.3, 0.3, 0.3),
                camera_fov=40 # angle look at
            ),

            vis_options=gs.options.VisOptions(
                # rendered_envs_idx=list(range(1)),
                segmentation_level='link',
                show_world_frame=False,
                env_separate_rigid = env_separate,
            ),

            rigid_options=gs.options.RigidOptions(
                constraint_solver=gs.constraint_solver.CG,
                enable_collision=True,
                enable_joint_limit=True,
                dt=1./self.sim_hz
            ),
            # renderer=gs.options.(
            # )
            show_viewer=show_interact_viewer,
        )

        self.plane : gs.engine.entities.RigidEntity = self.scene.add_entity(
            gs.morphs.URDF(
                file=self.path['plane'],
                fixed=True,
                links_to_keep=[
                    'marker',
                ]
            ),
            material=gs.materials.Rigid(gravity_compensation=1.0)
        )

        self.robot : gs.engine.entities.RigidEntity = self.scene.add_entity(
            gs.morphs.URDF(
                pos = (0, 0, 0.17),
                file = self.path['robot'],
                fixed=True,
                collision=True,
                links_to_keep=[
                    'tcp',
                    "flange_with_ori"
                ]
            )
        )

        self.cube : gs.engine.entities.RigidEntity = self.scene.add_entity(
            gs.morphs.URDF(
                pos = (0, 0, 0),
                file=self.path['TCube'],
                collision=True,
                fixed=True,
                visualization=True,
                links_to_keep=[
                    'center',
                ]
                )
        )
        
        # the top view only for evaling usage when we need to capture the overall view of the robotic arm and the Tcube workspace
        
        # self.cam = self.scene.add_camera(
        #     res=self.render_size,
        #     pos=(-0.3, 0.3, 0.8),
        #     lookat=(-0.3, 0.3, 0),
        #     fov=65,
        #     GUI=show_camera,
        # )

        # Original tilted view from one side of the workspace which is used for the robot observing the environment 
        self.cam = self.scene.add_camera(
            res=self.render_size,
            pos=(0, 0.3, 0.9),
            lookat=(-0.4, 0.3, 0),
            fov=65,
            GUI=show_camera,
        )
        # low level cam view 
        # self.cam = self.scene.add_camera(
        #     res=self.render_size,
        #     pos=(-1, 1, 0.5),
        #     lookat=(-0.3, 0.3, 0.35),
        #     fov=65,
        #     GUI=show_camera,
        # )

        # high level cam view
        # self.cam = self.scene.add_camera(
        #     res=self.render_size,
        #     pos=(-1.2, 1.2, 1),
        #     lookat=(-0.3, 0.3, 0.35),
        #     fov=65,
        #     GUI=show_camera,
        # )

        # the tested attached cameras which could be attached to the eef when during the rolling out, but the effect is not good for our eef which is short
        # self.cam_attached = self.scene.add_camera(
        #     res=self.render_size, 
        #     GUI=show_camera,
        #     fov=65,
        # )
        
        self.scene.build(n_envs=n_envs)

        jnt_names = [ 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 
                      'joint6', 'joint7', 'finger_width_joint'] 

        self.robot_dofs_idx = [self.robot.get_joint(name).dof_idx_local for name in jnt_names]
        self.cube_dofs_idx  = self.cube.get_joint('cube_plane_joint').dof_idx_local
        self.tcp: gs.engine.entities.rigid_entity.RigidLink = self.robot.get_link('tcp')
        self.eef: gs.engine.entities.rigid_entity.RigidLink = self.robot.get_link('flange_with_ori') 
        self.gripper: gs.engine.entities.rigid_entity_RigidJoint = self.robot.get_joint('finger_width_joint')  
        self.gripper_idx = self.gripper.dof_idx_local      
        self.eef_idx = self.eef.idx_local
        self.tcp_idx = self.tcp.idx_local

        # link's idx_local respect to its parent object
        # self.marker_dofs = self.plane.get_joint('marker_joint')
        self.marker_idx = self.plane.get_link('marker').idx_local
        self.marker_center_dof_idx = self.plane.get_joint('marker_joint').dof_idx_local
        self.Tcube_idx = self.cube.get_link('center').idx_local

        # offset_T = torch.Tensor(
        #     [[1, 0, 0, -0.1],
        #      [0, 1, 0, 0], 
        #      [0, 0, -1, -0.09], 
        #      [0, 0, 0, 1]]
        # )
        # self.cam_attached.attach(rigid_link = self.eef, offset_T = offset_T)

        self.robot.set_dofs_kp(
            kp             = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100]),
            dofs_idx_local = self.robot_dofs_idx,
        )
        self.robot.set_dofs_kv(
            kv             = np.array([450, 450, 350, 350, 200, 200, 200, 10]),
            dofs_idx_local = self.robot_dofs_idx,
        )
        self.render_cache = None
        self.seed()
        self.device = gs.device

    def seed(self, seed=None):
        if self._seed is None: # generate system level seed
            self._seed = np.random.randint(0, 25536)
            self.np_random_generators = None
        if seed is not None:
            if self.n_envs is None: raise ValueError("Envs have not been initialized!")
            if len(seed) != self.n_envs: raise ValueError("Given seed length is not" \
            "compatible with n_envs!")
            self.env_seed = seed
            self.np_random_generators = [np.random.default_rng(s) for s in seed]


    def reset_idx(self, envs_idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            envs_idx: torch.Tensor
        Returns:
            observation: Dict with torch tensors
        """
        num_reset = envs_idx.shape[0]
        if num_reset == 0:
            return
    
        block_positions: List[List[float]] = []
        target_positions: List[List[float]] = []
        block_angles: List[float] = []
        target_angles: List[float] = []
        
        for env_idx in envs_idx:
            env_idx_int = int(env_idx)  
            
            if hasattr(self, 'np_random_generators'):
                rng = self.np_random_generators[env_idx_int]
            else:
                raise ValueError("ENV-LEVEL seeds have not been defined!")
            
            x_center, y_center = self.spawn_center
            x_span = float(self.block_lim['xlim']) * self.spawn_range_scale
            y_span = float(self.block_lim['ylim']) * self.spawn_range_scale
            min_xy_sep = 0.12                        # meters, this effect the model performance greatly
            min_ang_sep = np.deg2rad(20.0)           # could set to zero 
            max_tries = 64

            def sample_xy():
                # Sample uniformly in region: [center - span, center + span]
                x = x_center - x_span + rng.random() * (2.0 * x_span)
                y = y_center - y_span + rng.random() * (2.0 * y_span)
                return x, y

            block_x, block_y = sample_xy()
            block_z = 0.06  # Fixed height
            block_angle = rng.random() * 2.0 * np.pi   # [0, 2π)

            # Sample target with min XY separation
            for _ in range(max_tries):
                target_x, target_y = sample_xy()
                if np.hypot(target_x - block_x, target_y - block_y) >= min_xy_sep:
                    break
            else:
                # Fallback: push target away along vector to band center
                dx, dy = (x_center - block_x), (y_center - block_y)
                n = np.hypot(dx, dy) + 1e-6
                target_x = block_x + (dx / n) * min_xy_sep
                target_y = block_y + (dy / n) * min_xy_sep

            target_z = 0.0  # Marker height
            target_angle = rng.random() * 2.0 * np.pi  # [0, 2π)

            # Enforce angular separation (optional)
            if min_ang_sep > 0:
                for _ in range(max_tries):
                    ang_diff = np.arctan2(np.sin(target_angle - block_angle), np.cos(target_angle - block_angle))
                    if np.abs(ang_diff) >= min_ang_sep:
                        break
                    target_angle = rng.random() * 2.0 * np.pi

            # Assign
            block_positions.append([block_x, block_y, block_z])
            target_positions.append([target_x, target_y, target_z])
            block_angles.append(block_angle)
            target_angles.append(target_angle)
    
        block_pos = torch.tensor(block_positions, device=gs.device, dtype=torch.float32)
        target_pos = torch.tensor(target_positions, device=gs.device, dtype=torch.float32)
        block_angle = torch.tensor(block_angles, device=gs.device, dtype=torch.float32).reshape(-1, 1)
        target_angle = torch.tensor(target_angles, device=gs.device, dtype=torch.float32).reshape(-1, 1)
         
        # save initial dis and ang for reward calculation baseline
        self.ini_delta_dis = torch.linalg.norm(block_pos[:, :2] - target_pos[:, :2], axis=1, keepdims=False)
        # Wrap angle difference to [-pi, pi]
        ang_diff0 = torch.atan2(torch.sin(block_angle - target_angle), torch.cos(block_angle - target_angle))
        self.ini_delta_ang = torch.abs(ang_diff0).squeeze()

        # Calculate baseline reward (reward at initial state)
        # Position component at initial distance
        max_distance = 0.5
        pos_reward_init = 5.0 * torch.exp(-3.0 * self.ini_delta_dis / max_distance)
        # Angle component at initial angle difference
        ang_reward_init = 2.0 * torch.cos(self.ini_delta_ang)
        ang_reward_init = torch.clamp(ang_reward_init, 0.0, 2.0)
        # Baseline = position + angle (no success bonus, no progress bonus at t=0)
        self.reward0 = pos_reward_init + ang_reward_init

        block_state_torch = torch.concatenate([
            block_pos,
            torch.zeros(num_reset, 2, dtype=torch.float32, device=gs.device),  # roll & pitch 0
            block_angle
        ], axis=-1)
        # print(block_state_torch[0])

        target_state_torch = torch.concatenate([
            target_pos,
            torch.zeros(num_reset, 2, dtype=torch.float32, device=gs.device),  # roll & pitch 0
            target_angle
        ], axis=-1)

        # the original home_pos used for the collecting data and first version pushT
        home_pos_down = torch.tensor([-0.4602,  1.3013,  2.5882,  1.1296,  0.7087,  0.6787,  1.2742], device=gs.device).repeat(len(envs_idx), 1)
        
        # home pos position at [-0.7, 0.7, 0.6] when the robot base link is on the ground
        # home_pos_down = torch.tensor([2.5575,  -1.6492,  0.0000, -1.4629,  0.0000,  -1.3845,  1.8101], device=gs.device).repeat(len(envs_idx), 1)
        # home_pos_down_full = torch.cat([home_pos_down, torch.zeros(len(envs_idx), 7)],
        #                                 dim=1)

        # home_pos_down = self._ikine(self.eef, 
        #                             pos=torch.tensor([-0.4, 0.4, 0.4]).repeat(num_reset, 1),
        #                             quat=torch.tensor([0, 0, 1, 0]).repeat(num_reset, 1), 
        #                             envs_idx = envs_idx, 
        #                             init_qpos=None)[:, 0:7]
                    
        self.home_pos = home_pos_down
        # home_pos_down = torch.tensor([-0.3794,  1.3189,  2.6545,  1.5788,  1.0798,  1.0309,  0.8671]).repeat(self.n_envs, 1)

        # home_pos_down = torch.Tensor([[ 2.5060, -1.5572, -0.5973,  2.4180,  2.3973,  0.5915, -0.9210]]).repeat(self.n_envs, 1)
        # The pose which is controlling the bot to the height of 0.3 meters(which is a bit height when the robot is set to 0.3 meters high)
        
        # home_pos_down = torch.tensor([-2.7925,  1.6561, -0.0733, -1.7983, -0.2336,  1.2497,  2.9667], device=gs.device, 
        #                              dtype=torch.float32).repeat(self.n_envs, 1)
        
        # print("Predicted joint position:", home_pos_down)
        # raise ValueError # used for print Predicted joint position not being hided from other messages 
    
        self.robot.set_dofs_position(
            position=home_pos_down,
            dofs_idx_local=self.robot_dofs_idx[0:7],
            envs_idx=envs_idx
        )
        
        self.cube.set_dofs_position(
            block_state_torch,
            dofs_idx_local=self.cube_dofs_idx,
            envs_idx=envs_idx
        )
        
        self.plane.set_dofs_position(
            target_state_torch,
            dofs_idx_local=self.marker_center_dof_idx,
            envs_idx=envs_idx
        )
        self.robot.control_dofs_position(
            home_pos_down,
            dofs_idx_local=self.robot_dofs_idx[0:7],
            envs_idx=envs_idx
        ) # stablize initial pos(preventing falling down mistakenly)
     
        observation = self._get_obs(rgb=True, envs_idx=envs_idx)
        return observation
    
    def reset(self) -> Dict[str, torch.Tensor]:
        envs_idx_torch = torch.arange(self.n_envs, device=gs.device, dtype=torch.int32)
        return self.reset_idx(envs_idx=envs_idx_torch)
        

    def step(self, action: Optional[torch.Tensor] = None, 
             envs_idx: Optional[torch.Tensor] = None, 
             cal_all_keypoints: bool = False) -> Tuple[Dict[str, torch.Tensor], Union[float, np.ndarray], List[bool], Dict[str, torch.Tensor]]:
        
        if envs_idx is None:
            envs_idx = torch.arange(self.n_envs, device=gs.device, dtype=torch.int32)

        # Always maintain full environment indexing for consistent tensor shapes
        all_envs_idx = torch.arange(self.n_envs, device=gs.device, dtype=torch.int32)

        
        n_steps = int(self.sim_hz // self.control_hz)
        if action is not None:
            shape = np.shape(action)
            if shape[0] != len(envs_idx): 
                raise ValueError("Action_dims are not compatible with n_envs")
            else: 
               quat = torch.tile(torch.tensor([0, 0, 1, 0], device=self.device), (len(envs_idx), 1))
               qpos = self._ikine(self.eef, action, quat, envs_idx)
            #  print(qpos)
            self.robot.control_dofs_position(position=qpos[:, 0:7], # does not control tcp joints
                                        dofs_idx_local=self.robot_dofs_idx[0:7], 
                                        envs_idx=envs_idx
                                        )
            # control the eef to maintain closing during each step to make sure it will not open when execution
            self.robot.control_dofs_position(position = torch.Tensor([0.0]).repeat((len(envs_idx), 1)), dofs_idx_local = self.gripper_idx, envs_idx = envs_idx)
    
            # When the action predicted is smooth, no need for using plan for executing the action sequence
            # waypoints = self.robot.plan_path(
            #        qpos_goal = qpos, 
            #        num_waypoints = n_steps
            # )

        sim_steps = 0
        for _ in range(n_steps):
            self.scene.step()
            if sim_steps % int(self.step_per_render) == 0: # make sure that the frames being rendered is at about 30 fps
                # Always get observations for ALL environments to maintain consistent indexing
                observation = self._get_obs(rgb=True, envs_idx=all_envs_idx) # * render less for faster simulation 
            sim_steps = sim_steps + 1
        # for point in waypoints:
        #     self.robot.control_dofs_position(position=point[:, 0:7], # does not control tcp joints
        #                                 dofs_idx_local=self.robot_dofs_idx[0:7], 
        #                                 envs_idx=envs_idx
        #                                 )   
        #     self.scene.step()

        # Always get poses for ALL environments to maintain consistent indexing
        self._get_poses(all_envs_idx) 
        self.calculate_all_keypoints()

        # print("eef pose: ", self.poses['agent_pos'][0,:2])
        # print("cube pose:", self.poses['cur_Tpose'][0,:])
        
        info = self._get_info(all_envs_idx) # calculate the info for the all envs for convenience 
        
        done = [ratio > self.done_ratio for ratio in self._cal_intersection()]
        reward = self._cal_rewards()

        return observation, reward, done, info    
    
    def render(self, mode='rgb_array'):
        assert mode == 'rgb_array'
        self._get_obs(segmentation=False)
        if self.render_cache is None:
            self._get_obs()
        return self.render_cache
    
    
    def start_recording(self, ):
        assert self.cam is not None
        self.cam.start_recording()
    
    def stop_recording(self, filename=None):
        # target_folder is configured in env_config
        # if filename is None:
        #     os.makedirs(target_folder, exist_ok=True)
        #     filename = os.path.join(target_folder, time.strftime("%Y%m%d-%H-%M") + "-pushT-env.mp4")
        
        self.cam.stop_recording(save_to_filename=filename, fps=self.fps)
        
        # os.chdir(old_dir)


    def calculate_all_keypoints(self) -> None:
        # dis_ori = R_z * R_y * R_x * dis_direct

        if self.poses is None: 
            all_envs = torch.arange(self.n_envs, device=gs.device, dtype=torch.int32)
            self._get_poses(all_envs)
        
        cur_rot_ang_z = self.poses['cur_Tpose'][:, 5]
        tar_rot_ang_z = self.poses['target_Tpose'][:, 5]


        dis_direct = torch.tensor([
            [[-0.03, -0.1, 0], [0.03, -0.1, 0], [0.03, -0.03, 0],
             [0.23, -0.03, 0], [0.23, 0.03, 0], [0.03, 0.03, 0],
             [0.03, 0.1, 0], [-0.03, 0.1, 0]]
            for _ in range(self.n_envs)], device=gs.device)

        # Using row vector for saving the relative distance, so taking multiplication on the right side to
        # multiply with unit vectors.
        def Rz(angz):
            cos_z = torch.cos(angz)
            sin_z = torch.sin(angz)
            zeros = torch.zeros_like(angz)
            ones = torch.ones_like(angz)

            return torch.transpose(torch.stack([
                torch.stack([cos_z, -sin_z, zeros], dim=-1),
                torch.stack([sin_z, cos_z, zeros], dim=-1),
                torch.stack([zeros, zeros, ones], dim=-1)
            ], dim=-2), dim0=-2, dim1=-1)
        
        # print(dis_direct.shape)
        # print(cur_rot_ang_z.shape)
        # print(Rz(cur_rot_ang_z).shape)                  
        cur_ori_dis = dis_direct @ Rz(cur_rot_ang_z)
        target_ori_dis = dis_direct @ Rz(tar_rot_ang_z)


        cur_center = self.poses['cur_Tpose'][:, 0:3].reshape(-1, 1, 3)
        target_center = self.poses['target_Tpose'][:, 0:3].reshape(-1, 1, 3)

        self.keypoints = {
            'cur_keypoints': cur_center + cur_ori_dis,      # torch Tensor
            'target_keypoints': target_center + target_ori_dis  # torch Tensor
        }
    

    def _cal_rewards(self) -> Union[torch.FloatType, torch.Tensor]:
        """
        Baseline-centered interpretable reward function for PushT task.

        Reward components:
        1. Position proximity: [0, 5] - Higher when object closer to target
        2. Angle alignment: [0, 2] - Higher when angles match
        3. Success bonus: +10 - Large bonus for task completion
        4. Progress bonus: [0, 1] - Rewards improvement from initial state

        Returns: reward - baseline (centered around 0)
        - Negative values: Worse than initial position
        - ~0: Similar to initial position
        - Positive values: Better than initial position
        - Large positive (>10): Task completion
        """
        if self.poses is None:
            all_envs = torch.arange(self.n_envs, device=gs.device, dtype=torch.int32)
            self._get_poses(all_envs)

        cur_pos = self.poses['cur_Tpose']
        tar_pos = self.poses['target_Tpose']

        # Calculate current distance and angle errors
        pos_error = torch.linalg.norm(cur_pos[:, :2] - tar_pos[:, :2], dim=1)
        ang_diff = torch.atan2(torch.sin(cur_pos[:, 5] - tar_pos[:, 5]), torch.cos(cur_pos[:, 5] - tar_pos[:, 5]))
        ang_error = torch.abs(ang_diff)

        # === COMPONENT 1: Position Proximity Reward (0-5 points) ===
        max_distance = 0.5  # meters (reasonable workspace)
        pos_reward = 5.0 * torch.exp(-3.0 * pos_error / max_distance)

        # === COMPONENT 2: Angle Alignment Reward (0-2 points) ===
        ang_reward = 2.0 * torch.cos(ang_error)
        ang_reward = torch.clamp(ang_reward, 0.0, 2.0)

        # === COMPONENT 3: Success Bonus (0 or 10 points) ===
        intersection_ratios = torch.tensor(self._cal_intersection(), device=gs.device)
        success_bonus = 10.0 * (intersection_ratios > self.done_ratio).float()

        # === COMPONENT 4: Progress Bonus (0-1 points) ===
        pos_progress = torch.clamp((self.ini_delta_dis - pos_error) / self.ini_delta_dis, 0.0, 1.0)

        # === RAW REWARD ===
        raw_reward = pos_reward + ang_reward + success_bonus + pos_progress

        # === BASELINE-CENTERED REWARD ===
        # Subtract the baseline reward calculated at reset
        centered_reward = raw_reward - self.reward0

        # Optional: Add debugging info (can be disabled for training)
        if hasattr(self, 'debug_rewards') and self.debug_rewards:
            print(f"Reward breakdown - Pos: {pos_reward[0]:.2f}, Ang: {ang_reward[0]:.2f}, "
                  f"Success: {success_bonus[0]:.2f}, Progress: {pos_progress[0]:.2f}, "
                  f"Raw: {raw_reward[0]:.2f}, Baseline: {self.reward0[0]:.2f}, "
                  f"Centered: {centered_reward[0]:.2f}")

        return centered_reward


    def _ikine(self, link: gs.engine.entities.rigid_entity.RigidLink, 
               pos: torch.Tensor, quat: torch.Tensor, 
               envs_idx: torch.Tensor, init_qpos=None) -> torch.Tensor: 
        

        if init_qpos is None: init_qpos = self.robot.get_dofs_position(envs_idx = envs_idx)
        # control_idx = [0, 1, 3, 4, 5, 6]
        control_idx = list(range(7))
        qpos = self.robot.inverse_kinematics(
            link=link,
            init_qpos=init_qpos, 
            pos=pos,  
            quat=quat, 
            dofs_idx_local=[self.robot_dofs_idx[idx] for idx in control_idx],
            envs_idx=envs_idx, 
            # respect_joint_limit=False
        )
        # print(qpos)
        return qpos
    

    def _cal_intersection(self) -> List[float]:

        if self.keypoints is None:
            return []
                    
        cur_points_np = to_numpy(self.keypoints['cur_keypoints'][:, :, :2])
        target_points_np = to_numpy(self.keypoints['target_keypoints'][:, :, :2])
        ratio: List[float] = []
        for i in range(cur_points_np.shape[0]):
            try:

                cur_points = cur_points_np[i]
                tar_points = target_points_np[i]

                if not np.all(np.isfinite(cur_points)) or not np.all(np.isfinite(tar_points)):
                    print(f"Env {i}: Invalid values detected")
                    ratio.append(0.0)
                    continue

                cur_polygon = Polygon(cur_points)
                tar_polygon = Polygon(tar_points)

                intersection_geom = cur_polygon.intersection(tar_polygon)
                area = intersection_geom.area
                ratio.append(area / cur_polygon.area if cur_polygon.area > 0 else 0.0)

            except Exception as e:
                raise ValueError(f"Env {i}: Polygon error - {e}")
                
        return ratio

    def _get_poses(self, envs_idx: torch.Tensor, global_write=True, ret=False) -> None:
        """Get poses from Genesis API with robust NaN detection and fallback."""

        assert envs_idx is not None

        # sometimes when the dofs initial pos 0 is not aligned with the urdf file the cooridate 0 which has a small bias, the value here using the dofs is not accurate 
        cur_Tpose_torch = self.cube.get_dofs_position(self.cube_dofs_idx, envs_idx)
        target_Tpose_torch = self.plane.get_dofs_position(self.marker_center_dof_idx, envs_idx)
        agent_pos_torch = self.robot.get_links_pos(self.eef_idx, envs_idx)[:, 0, :]

        if global_write:
            self.poses = {
            'cur_Tpose': cur_Tpose_torch,      # torch tensor in gs.device
            'target_Tpose': target_Tpose_torch, # torch tensor in gs.device
            'agent_pos': agent_pos_torch        # torch tensor in gs.device
            }
        
        if ret: 
            return self.poses


    def _get_obs(self, rgb: bool = True, depth: bool = False, 
                 segmentation: bool = False, normal: bool = False, 
                 envs_idx: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        if envs_idx is None:
            envs_idx = torch.arange(self.n_envs, device=gs.device, dtype=torch.int32)
        elif not isinstance(envs_idx, torch.Tensor):
            raise ValueError("envs_idx must be torch.Tensor for GPU simulation API!")

        # img (list:[w, h, 3],NoneType,NoneType,NoneType)
        img = self.cam.render(rgb=rgb, depth=depth, segmentation=segmentation, normal=normal)
        # img_attached = self.cam_attached.render(rgb=rgb, depth=depth, segmentation=segmentation, normal=normal) # render the eef attached cam scene 

        idx_np = to_numpy(envs_idx, float=False)  # numpy indices for array indexing

        agent_pos_torch = self.robot.get_links_pos(self.eef_idx, None)

        """
        On mac, because we can't use env_separate option as the opengl version does not support mac, we only render one env one time, 
        and the returned image does not have the first dim showing the idx.

        On linux or windows machine that supports env_separate for rigid rendering, the video saving could be separated into multi envs with
        _idx, and the imgs is returned with the idx at the dim0 
        """ 
        if gs.device == torch.device('mps:0') or gs.device ==torch.device('mps'):  
            obs = {
                'envs_idx': envs_idx,  # keep as torch tensor
                'image': torch.unsqueeze(to_torch(img[0].copy()), dim=0),  # convert numpy to torch
                'agent_pos': agent_pos_torch[idx_np, 0, :2]  # torch tensor indexed with numpy
            }
        else:
            obs = {
                'envs_idx': envs_idx,  # keep as torch tensor
                'image': to_torch(img[0][idx_np, :]),  # convert numpy to torch
                'agent_pos': agent_pos_torch[idx_np, 0, :2]  # torch tensor indexed with numpy
            }

        return obs
    
    def _get_info(self, envs_idx: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:

        if envs_idx is None:
            envs_idx = torch.arange(self.n_envs, device=gs.device, dtype=torch.int32)
        elif not isinstance(envs_idx, torch.Tensor):
            raise ValueError("envs_idx must be torch.Tensor for GPU simulation API!")
        info = self._get_poses(envs_idx=envs_idx, global_write=False, ret=True).copy()
        info['envs_idx'] = envs_idx

        return info
    
if __name__ == '__main__':
    env = PushTEnv()
    env.start(n_envs=1, show_camera=False, show_interact_viewer=True, 
              env_separate=False, seed=[0])
    env.seed([0])
    env.reset()
    current_time = time.time()

    # env.start_recording()
    for i in range(5000):
        current_time = time.time()
        _, _, _, info = env.step(action=torch.tensor([[-0.4, 0.4, 0.27]]))
        # _, _, _, info = env.step()

        finishing_time = time.time()
        executing_time = finishing_time - current_time

        # print(info)
        # print(executing_time)

        time_to_wait = 0.1 - executing_time
        # if(time_to_wait > 0):
        #     time.sleep(time_to_wait)

        # env.robot.control_dofs_position(position = torch.Tensor([0.0]).repeat((env.n_envs, 1)), dofs_idx_local = env.gripper_idx, envs_idx = None)
        # if i % 10 == 0: 
            # env.reset()
            # print(env.calculate_all_keypoints())
            # print(np.shape(env.render()[0]))
            # print(env._cal_intersection())
            # print(env.get_key_points())

    # env.stop_recording()