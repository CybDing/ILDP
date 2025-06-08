import time
import os
import sys

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

from genesis_ILDP.utils.cuda import *
from genesis_ILDP.config.env_config import *
from shapely.geometry import Polygon

class PushTEnv(gym.Env):
    # TODO close(), 
    metadata = {"render.mode": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self,
                 render_size=(96, 96),
                 xlim=.3,
                 ylim=.3,
                 seed=None, # seed 
                 model_path=env_path,
                 fps = 30,
                 show_fps = True,
                 ):

        self.render_size = render_size
        self.sim_hz = 100
        self.control_hz = 10 # how long waiting for robotic arms to finsih exectuing an action
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

        self.ini_delta_dis = 0
        self.ini_delta_ang = 0 
        self.reward0 = None
        self.poses = None
        self.keypoints = None

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
        # init only once
        assert self.is_init == False 
        self.n_envs = n_envs
        
        gs.init(
            seed = 0,
            #seed = self._seed
            backend = gs.gpu
        )
        self.scene = gs.Scene(
            show_FPS=self.show_fps,
            sim_options=gs.options.SimOptions(dt=1./self.sim_hz, substeps=1),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(self.sim_hz),
                camera_pos=(2, 2, 1.5),
                camera_lookat=(0.7, 0.7, 0.3),
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
                pos = (0, 0, 0.3),
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
        # box_baselink_joint, box_baselink

        self.cam = self.scene.add_camera(
            res=self.render_size,
            pos=(1, 1, 0.7),
            lookat=(0.7, 0.7, 0.0),
            fov=60,
            GUI=show_camera,
        )
        
        self.scene.build(n_envs=n_envs)

        jnt_names = [ 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 
                      'joint6', 'joint7', 'finger_width_joint'] 

        self.robot_dofs_idx = [self.robot.get_joint(name).dof_idx_local for name in jnt_names]
        self.cube_dofs_idx  = self.cube.get_joint('cube_plane_joint').dof_idx_local
        self.tcp: gs.engine.entities.rigid_entity.RigidLink = self.robot.get_link('tcp')
        self.eef: gs.engine.entities.rigid_entity.RigidLink = self.robot.get_link('flange_with_ori') 
        self.eef_idx = self.eef.idx_local
        self.tcp_idx = self.tcp.idx_local

        self.marker_idx = self.plane.get_link('marker').idx_local
        self.marker_dofs_idx = self.plane.get_joint('marker_joint').dof_idx_local
        self.Tcube_idx = self.cube.get_link('center').idx_local

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

    # setting seed for generator
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


    def reset_idx(self, envs_idx: torch.Tensor):
        num_reset = envs_idx.shape[0]
        if num_reset == 0:
            return

        block_positions = []
        target_positions = []
        block_angles = []
        target_angles = []
        
        for i, env_idx in enumerate(envs_idx):
            env_idx = int(env_idx)
            
            if hasattr(self, 'np_random_generators'):
                rng = self.np_random_generators[env_idx]
            else:
                raise ValueError("ENV-LEVEL seeds have not been defined!")
            
            block_x = rng.random() * self.block_lim['xlim'] + 0.2
            block_y = rng.random() * self.block_lim['ylim'] + 0.2
            block_z = 0.05  # Fixed Height
            
            block_angle = rng.random() * np.pi / 2
            
            target_x = rng.random() * self.block_lim['xlim'] + 0.2
            target_y = rng.random() * self.block_lim['ylim'] + 0.2
            target_z = 0.0  # Marker at Ground level
            
            target_angle = rng.random() * np.pi - np.pi / 2
        
            block_positions.append([block_x, block_y, block_z])
            target_positions.append([target_x, target_y, target_z])
            block_angles.append(block_angle)
            target_angles.append(target_angle)
    
        block_pos = np.array(block_positions)
        target_pos = np.array(target_positions)
        block_angle = np.array(block_angles).reshape(-1, 1)
        target_angle = np.array(target_angles).reshape(-1, 1)

        self.ini_delta_dis = np.linalg.norm(block_pos[:, :2] - target_pos[:, :2], axis=1, keepdims=False)
        self.ini_delta_ang = np.abs(block_angle[:]- target_angle[:])
        self.reward0 = 1 / (1 + 0.1) * 1 / np.sqrt(1 + 0.1)
        
        block_state = to_torch(np.concatenate([
            block_pos,
            np.zeros(shape=(num_reset, 2)),  # roll & pitch 保持为0
            block_angle
        ], axis=-1))
        
        target_state = to_torch(np.concatenate([
            target_pos,
            np.zeros(shape=(num_reset, 2)),  # roll & pitch 保持为0
            target_angle
        ], axis=-1))
        
        # home_pos = torch.zeros(size=(num_reset, len(self.robot_dofs_idx)), device=gs.device)
        home_pos_down = self._ikine(self.eef, 
                                    pos=torch.tensor([0.1, 0.1, 0.3]).repeat(num_reset, 1),
                                    quat=torch.tensor([0, 0, 1, 0]).repeat(num_reset, 1), 
                                    envs_idx = envs_idx)[:, 0:7]

        self.robot.set_dofs_position(
            position=home_pos_down,
            dofs_idx_local=self.robot_dofs_idx[0:7],
            envs_idx=envs_idx
        )
        
        self.cube.set_dofs_position(
            block_state,
            dofs_idx_local=self.cube_dofs_idx,
            envs_idx=envs_idx
        )
        
        self.plane.set_dofs_position(
            target_state,
            dofs_idx_local=self.marker_dofs_idx,
            envs_idx=envs_idx
        )
        self.robot.control_dofs_position(
            home_pos_down,
            dofs_idx_local=self.robot_dofs_idx[0:7],
            envs_idx=envs_idx
        ) # stableize initial pos(preventing falling down mistakenly)

        ## TODO control eef to close(1)
        
        observation = self._get_obs(rgb=True, envs_idx=envs_idx)
        return observation

    

    def reset(self,):
        return self.reset_idx(envs_idx=torch.arange(self.n_envs, device=gs.device, dtype=torch.int16)
    )
        
    def step(self, action=None, envs_idx = None, cal_all_keypoints=False):
        # action: agent_pos(eef_pos) n_envs * action_x, action_y
        # one action, multi sim
        if envs_idx is None:
            envs_idx = torch.arange(self.n_envs)
        n_steps = int(self.sim_hz // self.control_hz)
        if action is not None:
            shape = np.shape(action)
            if shape[0] != len(envs_idx): 
                raise ValueError("Action_dims are not compatible with n_envs")
            else: 
               quat = torch.tile(torch.tensor([0, 0, 1, 0], device=self.device), (len(envs_idx), 1))
               qpos = self._ikine(self.eef, action, quat, envs_idx)
               self.robot.control_dofs_position(position=qpos[:, 0:7], # does not control tcp joints
                                        dofs_idx_local=self.robot_dofs_idx[0:7], 
                                        envs_idx=envs_idx
                                     ) 
        for _ in range(n_steps):
            self.scene.step()

        self._get_poses(envs_idx) # get Tpos, agent_pos
        self.calculate_all_keypoints()
        print(self.poses['agent_pos'][0,:2])
        ### JUDGE after sim steps, preventing misjudge the done condition
        observation = self._get_obs(rgb=True, envs_idx=envs_idx)
        info = self._get_info(envs_idx)
        
        done = [ratio > 0.6 for ratio in self._cal_intersection()]
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
        os.makedirs(target_folder, exist_ok=True)
        old_dir = os.getcwd()
        os.chdir(target_folder)

        if filename is None:
            filename = os.path.join(target_folder, time.strftime("%Y%m%d-%H-%M") + "-pushT-env.mp4")
        self.cam.stop_recording(save_to_filename=filename, fps=self.fps)
        
        os.chdir(old_dir)

    
    def calculate_all_keypoints(self, ):
        # dis_ori = R_z * R_y * R_x * dis_direct

        if self.poses is None: self._get_poses()
        poses = self.poses.copy()
        cur_rot_ang_z = to_numpy(poses['cur_Tpose'][:,5])
        tar_rot_ang_z = to_numpy(poses['target_Tpose'][:,5])

        dis_direct = np.array([[-0.03, -0.1, 0], [0.03, -0.1, 0], [0.03, -0.03, 0],
                                [0.23, -0.03, 0], [0.23, 0.03, 0], [0.03, 0.03, 0], 
                                [0.03, 0.1, 0], [-0.03, 0.1, 0]])

        Rz = lambda angz: np.array([np.transpose(
                                np.array([[np.cos(ang_z), -np.sin(ang_z), 0], 
                                        [np.sin(ang_z),   np.cos(ang_z), 0 ],
                                        [0,              0,            1]], 
                                )) for ang_z in angz ])
                          
        cur_ori_dis = dis_direct @ Rz(cur_rot_ang_z)
        target_ori_dis = dis_direct @ Rz(tar_rot_ang_z)

        cur_center = to_numpy(poses['cur_Tpose'][:,0:3])
        target_center = to_numpy(poses['target_Tpose'][:,0:3])

        self.keypoints = {
            'cur_keypoints': cur_center + cur_ori_dis,  # broadcasting
            'target_keypoints': target_center + target_ori_dis
        }
    

    def _cal_rewards(self, ):
        if self.poses is None: self._get_poses()
        cur_pos = to_numpy(self.poses['cur_Tpose'])
        tar_pos = to_numpy(self.poses['target_Tpose'])

        dis = np.linalg.norm(cur_pos[:, :2]-tar_pos[:, :2])
        ang = np.abs(cur_pos[:, 5]- tar_pos[:, 5])

        return 1 / (dis/self.ini_delta_dis + 0.1) / np.sqrt(ang/self.ini_delta_ang + 0.1) - self.reward0


    def _ikine(self, link, pos, quat, envs_idx): 
        qpos = self.robot.inverse_kinematics(
        link = link,
        pos = pos,
        quat = quat,
        dofs_idx_local=self.robot_dofs_idx[0:7],
        envs_idx=envs_idx
       )
        return qpos
    

    def _cal_intersection(self, ):
        points = self.keypoints
        cur_points = points['cur_keypoints'][:,:,:2]
        target_points = points['target_keypoints'][:,:,:2]

        ratio = []
        for i in range(cur_points.shape[0]):
            cur_polygon = Polygon(cur_points[i])
            tar_Polygon = Polygon(target_points[i])

            intersection_geom = cur_polygon.intersection(tar_Polygon)
            area = intersection_geom.area 
            ratio.append(area / cur_polygon.area)

        return ratio

    def _get_poses(self, envs_idx:torch.Tensor):

        # return the first env's poses(pos+ang(xyz euler))

        cur_Tpose2=self.cube.get_dofs_position(self.cube_dofs_idx, envs_idx)
        target_Tpose2 = self.plane.get_dofs_position(self.marker_dofs_idx, envs_idx)
        self.poses = {
            'cur_Tpose': cur_Tpose2,
            'target_Tpose': target_Tpose2,
            'agent_pos': self.robot.get_links_pos(self.eef_idx, envs_idx)[:, 0, :2]
        }


    def _get_obs(self, rgb=True, depth=False, segmentation=False, normal=False, envs_idx=None):

        if envs_idx is None:
            envs_idx = torch.arange(self.n_envs, device=gs.device, dtype=torch.int16)
        else:
            if isinstance(envs_idx, torch.Tensor) == False:
                raise ValueError("Dtype for envs_idx is Torch Tensor!")

        # img (list:[w, h, 3],NoneType,NoneType,NoneType)
        img = self.cam.render(rgb=rgb, depth=depth, segmentation=segmentation, normal=normal) 
        self.render_cache = img

        # jnt_pos = [self.robot.get_dofs_position(idx, envs_idx=np.arange(self.n_envs)) for idx in self.dofs_idx]
        agent_pos = self.robot.get_links_pos(self.eef_idx, envs_idx)
        idx = to_numpy(envs_idx, float=False)
        obs = {
            'envs_idx': envs_idx,
            'image': to_torch(img[0][idx,:]),
            'agent_pos': agent_pos[idx,0, :2]
        }
        # marker_pos = self.plane.get_links_pos(self.marker_idx, envs_idx)     
        return obs
    
    def _get_info(self, envs_idx = None):
        if envs_idx is None:
            envs_idx = torch.arange(self.n_envs, device=gs.device)
        else:
            if isinstance(envs_idx, torch.Tensor) == False:
                raise ValueError("Dtype for envs_idx is Torch Tensor!")
        idx = to_numpy(envs_idx, float=False) 
        info = {
            'envs_idx': envs_idx,
            "agent_pos": self.eef.get_pos()[idx, :2],
            "goal_pos": self.plane.get_links_pos(self.marker_idx)[idx, :2],
        }
        return info
    

if __name__ == '__main__':
    env = PushTEnv()
    env.start(n_envs=1, show_camera=False, show_interact_viewer=False, 
              env_separate=False, seed=[0])
    env.seed(np.arange(1))
    env.reset()

   # env.start_recording()
    for i in range(500):
        # env.step(action=torch.tensor([[0.1, 0.1],
        #                               [0.1, 0.1]]))
        env.step()
        if i % 10 == 0: 
            # env.reset()
            # print(env.calculate_all_keypoints())
            print(np.shape(env.render()[0]))
            # print(env._cal_intersection())
            # print(env.get_key_points())

    #env.stop_recording()