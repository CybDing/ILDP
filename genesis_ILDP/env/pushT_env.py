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

from ..utils.cuda import *
from ..config.env_config import *
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

        # 环境状态变量 (numpy arrays for computation)
        self.ini_delta_dis: Union[float, np.ndarray] = 0.0
        self.ini_delta_ang: Union[float, np.ndarray] = 0.0 
        self.reward0: Optional[float] = None
        self.poses: Optional[Dict[str, torch.Tensor]] = None  # torch tensors from Genesis API
        self.keypoints: Optional[Dict[str, np.ndarray]] = None  # numpy arrays for computation

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


    def reset_idx(self, envs_idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        """重置指定环境的状态
        Args:
            envs_idx: torch.Tensor - 需要重置的环境索引 (GPU tensor for Genesis API)
        Returns:
            observation: Dict with torch tensors
        """
        num_reset = envs_idx.shape[0]
        if num_reset == 0:
            return

        # 使用list收集数据，最后转换为numpy再转torch
        block_positions: List[List[float]] = []
        target_positions: List[List[float]] = []
        block_angles: List[float] = []
        target_angles: List[float] = []
        
        for env_idx in envs_idx:
            env_idx_int = int(env_idx)  # 转为int用于numpy索引
            
            if hasattr(self, 'np_random_generators'):
                rng = self.np_random_generators[env_idx_int]
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
    
        # numpy arrays for computation
        block_pos_np = np.array(block_positions, dtype=np.float32)
        target_pos_np = np.array(target_positions, dtype=np.float32)
        block_angle_np = np.array(block_angles, dtype=np.float32).reshape(-1, 1)
        target_angle_np = np.array(target_angles, dtype=np.float32).reshape(-1, 1)

        # 保存初始状态用于奖励计算 (numpy)
        self.ini_delta_dis = np.linalg.norm(block_pos_np[:, :2] - target_pos_np[:, :2], axis=1, keepdims=False)
        self.ini_delta_ang = np.abs(block_angle_np - target_angle_np)
        self.reward0 = 1 / (1 + 0.1) * 1 / np.sqrt(1 + 0.1)
        
        # 转换为torch tensor用于Genesis API
        block_state_torch = to_torch(np.concatenate([
            block_pos_np,
            np.zeros(shape=(num_reset, 2), dtype=np.float32),  # roll & pitch 保持为0
            block_angle_np
        ], axis=-1))
        
        target_state_torch = to_torch(np.concatenate([
            target_pos_np,
            np.zeros(shape=(num_reset, 2), dtype=np.float32),  # roll & pitch 保持为0
            target_angle_np
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
            block_state_torch,
            dofs_idx_local=self.cube_dofs_idx,
            envs_idx=envs_idx
        )
        
        self.plane.set_dofs_position(
            target_state_torch,
            dofs_idx_local=self.marker_dofs_idx,
            envs_idx=envs_idx
        )
        self.robot.control_dofs_position(
            home_pos_down,
            dofs_idx_local=self.robot_dofs_idx[0:7],
            envs_idx=envs_idx
        ) # stablize initial pos(preventing falling down mistakenly)

        # TODO control eef to close(1)
        
        observation = self._get_obs(rgb=True, envs_idx=envs_idx)
        return observation
    

    def reset(self) -> Dict[str, torch.Tensor]:
        envs_idx_torch = torch.arange(self.n_envs, device=gs.device, dtype=torch.int32)
        return self.reset_idx(envs_idx=envs_idx_torch)
        
    def step(self, action: Optional[torch.Tensor] = None, 
             envs_idx: Optional[torch.Tensor] = None, 
             cal_all_keypoints: bool = False) -> Tuple[Dict[str, torch.Tensor], Union[float, np.ndarray], List[bool], Dict[str, torch.Tensor]]:
        """执行环境步骤
        Args:
            action: torch.Tensor - 动作 (需要是torch tensor用于Genesis API)
            envs_idx: torch.Tensor - 环境索引 (GPU tensor)
        """
        # action: agent_pos(eef_pos) n_envs * action_x, action_y
        # one action, multi sim
        if envs_idx is None:
            envs_idx = torch.arange(self.n_envs, device=gs.device, dtype=torch.int32)
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

        print("eef pose: ", self.poses['agent_pos'][0,:2])

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

    
    def calculate_all_keypoints(self) -> None:
        """计算所有关键点 (使用numpy进行几何计算)"""
        # dis_ori = R_z * R_y * R_x * dis_direct

        if self.poses is None: 
            default_envs = torch.arange(self.n_envs, device=gs.device, dtype=torch.int32)
            self._get_poses(default_envs)
        
        # 转换为numpy进行几何计算
        cur_rot_ang_z = to_numpy(self.poses['cur_Tpose'][:, 5])
        tar_rot_ang_z = to_numpy(self.poses['target_Tpose'][:, 5])

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

        cur_center = to_numpy(self.poses['cur_Tpose'][:, 0:3])
        target_center = to_numpy(self.poses['target_Tpose'][:, 0:3])

        # 保存为numpy arrays用于后续几何计算
        self.keypoints = {
            'cur_keypoints': cur_center + cur_ori_dis,      # numpy array for geometry
            'target_keypoints': target_center + target_ori_dis  # numpy array for geometry
        }
    

    def _cal_rewards(self) -> Union[float, np.ndarray]:
        """计算奖励 (使用numpy进行数值计算)"""
        if self.poses is None: 
            default_envs = torch.arange(self.n_envs, device=gs.device, dtype=torch.int32)
            self._get_poses(default_envs)
            
        # 转换为numpy进行数值计算
        cur_pos_np = to_numpy(self.poses['cur_Tpose'])
        tar_pos_np = to_numpy(self.poses['target_Tpose'])

        dis = np.linalg.norm(cur_pos_np[:, :2] - tar_pos_np[:, :2])
        ang = np.abs(cur_pos_np[:, 5] - tar_pos_np[:, 5])

        return 1 / (dis/self.ini_delta_dis + 0.1) / np.sqrt(ang/self.ini_delta_ang + 0.1) - self.reward0


    def _ikine(self, link: gs.engine.entities.rigid_entity.RigidLink, 
               pos: torch.Tensor, quat: torch.Tensor, 
               envs_idx: torch.Tensor) -> torch.Tensor:
        """逆运动学计算 (Genesis API需要torch tensors)""" 
        qpos = self.robot.inverse_kinematics(
            link=link,
            pos=pos,  # torch tensor
            quat=quat,  # torch tensor  
            dofs_idx_local=self.robot_dofs_idx[0:7],
            envs_idx=envs_idx  # torch tensor
        )
        return qpos
    

    def _cal_intersection(self) -> List[float]:
        """计算当前和目标多边形的交集比率 (使用numpy进行几何计算)"""
        if self.keypoints is None:
            return []
            
        # numpy arrays用于几何计算
        cur_points_np = self.keypoints['cur_keypoints'][:, :, :2]
        target_points_np = self.keypoints['target_keypoints'][:, :, :2]

        ratio: List[float] = []
        for i in range(cur_points_np.shape[0]):
            cur_polygon = Polygon(cur_points_np[i])
            tar_polygon = Polygon(target_points_np[i])

            intersection_geom = cur_polygon.intersection(tar_polygon)
            area = intersection_geom.area 
            ratio.append(area / cur_polygon.area)

        return ratio

    def _get_poses(self, envs_idx: torch.Tensor) -> None:
        """获取cube、target和agent的位姿
        Args:
            envs_idx: torch.Tensor - 环境索引 (GPU tensor for Genesis API)
        """

        # 从Genesis API获取torch tensors
        cur_Tpose_torch = self.cube.get_dofs_position(self.cube_dofs_idx, envs_idx)
        target_Tpose_torch = self.plane.get_dofs_position(self.marker_dofs_idx, envs_idx)
        agent_pos_torch = self.robot.get_links_pos(self.eef_idx, envs_idx)[:, 0, :2]
        
        self.poses = {
            'cur_Tpose': cur_Tpose_torch,      # torch tensor from Genesis
            'target_Tpose': target_Tpose_torch, # torch tensor from Genesis  
            'agent_pos': agent_pos_torch        # torch tensor from Genesis
        }


    def _get_obs(self, rgb: bool = True, depth: bool = False, 
                 segmentation: bool = False, normal: bool = False, 
                 envs_idx: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """获取观测值
        Args:
            envs_idx: torch.Tensor - 环境索引 (GPU tensor for Genesis API)
        Returns:
            Dict with torch tensors
        """

        if envs_idx is None:
            envs_idx = torch.arange(self.n_envs, device=gs.device, dtype=torch.int32)
        elif not isinstance(envs_idx, torch.Tensor):
            raise ValueError("envs_idx must be torch.Tensor for GPU simulation API!")

        # img (list:[w, h, 3],NoneType,NoneType,NoneType)
        img = self.cam.render(rgb=rgb, depth=depth, segmentation=segmentation, normal=normal) 
        self.render_cache = img

        # jnt_pos = [self.robot.get_dofs_position(idx, envs_idx=np.arange(self.n_envs)) for idx in self.dofs_idx]
        agent_pos_torch = self.robot.get_links_pos(self.eef_idx, envs_idx)  # torch tensor from Genesis
        idx_np = to_numpy(envs_idx, float=False)  # numpy indices for array indexing
        obs = {
            'envs_idx': envs_idx,  # keep as torch tensor
            'image': to_torch(img[0][idx_np, :]),  # convert numpy to torch
            'agent_pos': agent_pos_torch[idx_np, 0, :2]  # torch tensor indexed with numpy
        }
        # marker_pos = self.plane.get_links_pos(self.marker_idx, envs_idx)     
        return obs
    
    def _get_info(self, envs_idx: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """获取信息字典
        Args:
            envs_idx: torch.Tensor - 环境索引 (GPU tensor)
        """
        if envs_idx is None:
            envs_idx = torch.arange(self.n_envs, device=gs.device, dtype=torch.int32)
        elif not isinstance(envs_idx, torch.Tensor):
            raise ValueError("envs_idx must be torch.Tensor for GPU simulation API!")
        idx_np = to_numpy(envs_idx, float=False)  # numpy indices for array indexing 
        info = {
            'envs_idx': envs_idx,  # keep as torch tensor
            "agent_pos": self.eef.get_pos()[idx_np, :2],  # torch tensor indexed with numpy
            "goal_pos": self.plane.get_links_pos(self.marker_idx)[idx_np, :2],  # torch tensor indexed with numpy
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