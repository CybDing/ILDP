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

from genesis_ILDP.utils.cuda import *
from genesis_ILDP.config.env_config import *
from shapely.geometry import Polygon

class PushTEnv(gym.Env):
    # TODO close(), 
    metadata = {"render.mode": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self,
                 render_size=(500, 500),
                 xlim=.3,
                 ylim=.3,
                 seed=None, # seed 
                 model_path=env_path,
                 fps = 30,
                 show_fps = True,
                 ):

        self.render_size = render_size
        self.sim_hz = 100.0 # TODO check what does this sim_hz control here 
        self.control_hz = 10.0 # how long waiting for robotic arms to finsih exectuing an action
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
            show_FPS=True,
            sim_options=gs.options.SimOptions(dt=1./self.sim_hz, substeps=1),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=40,
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
            pos=(0.85, 0.85, 1.2),
            lookat=(0.3, 0.3, 0.0),
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
        """
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
            block_z = 0.07  # Fixed Height
            
            block_angle = rng.random() * np.pi / 2
            
            target_x = rng.random() * self.block_lim['xlim'] + 0.2
            target_y = rng.random() * self.block_lim['ylim'] + 0.2
            target_z = 0.0  # Marker at Ground level
            
            target_angle = rng.random() * np.pi - np.pi / 2
        
            block_positions.append([block_x, block_y, block_z])
            target_positions.append([target_x, target_y, target_z])
            block_angles.append(block_angle)
            target_angles.append(target_angle)
    
        block_pos = to_torch(np.array(block_positions, dtype=np.float32))
        target_pos = to_torch(np.array(target_positions, dtype=np.float32))
        block_angle = to_torch(np.array(block_angles, dtype=np.float32).reshape(-1, 1))
        target_angle = to_torch(np.array(target_angles, dtype=np.float32).reshape(-1, 1))
        
        # save initial dis and ang for reward calculation baseline
        self.ini_delta_dis = torch.linalg.norm(block_pos[:, :2] - target_pos[:, :2], axis=1, keepdims=False)
        self.ini_delta_ang = torch.abs(block_angle - target_angle)
        self.reward0 = torch.tensor(1 / (1 + 0.1) * 1 / np.sqrt(1 + 0.1), device=gs.device, dtype=torch.float32)

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

        # Debug: Store reset positions for environments 2 and 29
        if hasattr(self, 'debug_reset_poses'):
            self.debug_reset_poses.update({
                int(env_idx): {
                    'block_state': block_state_torch[i].cpu().numpy(),
                    'target_state': target_state_torch[i].cpu().numpy()
                } for i, env_idx in enumerate(envs_idx) if int(env_idx) in [2, 29]
            })
        else:
            self.debug_reset_poses = {
                int(env_idx): {
                    'block_state': block_state_torch[i].cpu().numpy(),
                    'target_state': target_state_torch[i].cpu().numpy()
                } for i, env_idx in enumerate(envs_idx) if int(env_idx) in [2, 29]
            }
        
        # home_pos = torch.zeros(size=(num_reset, len(self.robot_dofs_idx)), device=gs.device)
        # print("current joint position:", self.robot.get_dofs_position(envs_idx=envs_idx))
        # home_pos_down = self._ikine(self.eef, 
        #                             pos=torch.tensor([0.1, 0.1, 0.3]).repeat(num_reset, 1),
        #                             quat=torch.tensor([0, 0, 1, 0]).repeat(num_reset, 1), 
        #                             envs_idx = envs_idx)[:, 0:7]
        home_pos_down = torch.Tensor([[ 2.5060, -1.5572, -0.5973,  2.4180,  2.3973,  0.5915, -0.9210]])
        # print("Predicted joint position:", home_pos_down)
        # raise ValueError
    
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
        """execute actions
        Args:
            action: torch.Tensor (torch tensor for Genesis API)
            envs_idx: torch.Tensor
        """
        # action: agent_pos(eef_pos) n_envs * action_x, action_y, action_z
        # a single action, multi steps for smooth pid control of the arm
        # TODO check here

        if envs_idx is None:
            envs_idx = torch.arange(self.n_envs, device=gs.device, dtype=torch.int32)
            # print(111)
        # print("envs_idx", envs_idx)
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
            #    waypoints = self.robot.plan_path(
            #        qpos_goal = qpos, 
            #        num_waypoints = n_steps
            #    )
        for _ in range(n_steps):
            self.scene.step()
        # for point in waypoints:
        #     self.robot.control_dofs_position(position=point[:, 0:7], # does not control tcp joints
        #                                 dofs_idx_local=self.robot_dofs_idx[0:7], 
        #                                 envs_idx=envs_idx
        #                                 )   
        #     self.scene.step()

        self._get_poses(envs_idx) # get Tpos, agent_pos
        self.calculate_all_keypoints()

        print("eef pose: ", self.poses['agent_pos'][0,:2])
        # print("cube pose:", self.poses['cur_Tpose'][0,:])

        ### JUDGE after sim steps, preventing misjudge the done condition
        observation = self._get_obs(rgb=True, envs_idx=envs_idx)
        info = self._get_info(envs_idx)
        
        done = [ratio > 0.95 for ratio in self._cal_intersection()]
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
        # dis_ori = R_z * R_y * R_x * dis_direct

        if self.poses is None: 
            default_envs = torch.arange(self.n_envs, device=gs.device, dtype=torch.int32)
            self._get_poses(default_envs)
        
        cur_rot_ang_z = self.poses['cur_Tpose'][:, 5]
        tar_rot_ang_z = self.poses['target_Tpose'][:, 5]

        # # Debug: Check for invalid rotation angles and print problematic poses
        # if not torch.all(torch.isfinite(cur_rot_ang_z)):
        #     invalid_cur = torch.where(~torch.isfinite(cur_rot_ang_z))[0]
        #     print(f"Invalid cur_rot_ang_z in envs: {invalid_cur.cpu().numpy()}")
        #     for env_idx in invalid_cur:
        #         env_int = int(env_idx)
        #         print(f"Env {env_int} cur_Tpose: {self.poses['cur_Tpose'][env_idx].cpu().numpy()}")
        #         if hasattr(self, 'debug_reset_poses') and env_int in self.debug_reset_poses:
        #             print(f"Env {env_int} was reset to block_state: {self.debug_reset_poses[env_int]['block_state']}")
        # if not torch.all(torch.isfinite(tar_rot_ang_z)):
        #     invalid_tar = torch.where(~torch.isfinite(tar_rot_ang_z))[0]
        #     print(f"Invalid tar_rot_ang_z in envs: {invalid_tar.cpu().numpy()}")
        #     for env_idx in invalid_tar:
        #         env_int = int(env_idx)
        #         print(f"Env {env_int} target_Tpose: {self.poses['target_Tpose'][env_idx].cpu().numpy()}")
        #         if hasattr(self, 'debug_reset_poses') and env_int in self.debug_reset_poses:
        #             print(f"Env {env_int} was reset to target_state: {self.debug_reset_poses[env_int]['target_state']}")

        dis_direct = torch.tensor([
            [[-0.03, -0.1, 0], [0.03, -0.1, 0], [0.03, -0.03, 0],
             [0.23, -0.03, 0], [0.23, 0.03, 0], [0.03, 0.03, 0],
             [0.03, 0.1, 0], [-0.03, 0.1, 0]]
            for _ in range(self.n_envs)], device=gs.device)

        # using row vector for saving the relative distance, so taking multiplication on the right side to
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

        # Debug: Check for invalid values after rotation
        # if not torch.all(torch.isfinite(cur_ori_dis)):
        #     invalid_cur_dis = torch.where(~torch.all(torch.isfinite(cur_ori_dis), dim=(1,2)))[0]
        #     print(f"Invalid cur_ori_dis in envs: {invalid_cur_dis.cpu().numpy()}")
        # if not torch.all(torch.isfinite(target_ori_dis)):
        #     invalid_tar_dis = torch.where(~torch.all(torch.isfinite(target_ori_dis), dim=(1,2)))[0]
        #     print(f"Invalid target_ori_dis in envs: {invalid_tar_dis.cpu().numpy()}")

        cur_center = self.poses['cur_Tpose'][:, 0:3].view(-1, 1, 3)
        target_center = self.poses['target_Tpose'][:, 0:3].view(-1, 1, 3)

        # # Debug: Check center positions
        # if not torch.all(torch.isfinite(cur_center)):
        #     invalid_cur_center = torch.where(~torch.all(torch.isfinite(cur_center), dim=(1,2)))[0]
        #     print(f"Invalid cur_center in envs: {invalid_cur_center.cpu().numpy()}")

        self.keypoints = {
            'cur_keypoints': cur_center + cur_ori_dis,      # torch Tensor
            'target_keypoints': target_center + target_ori_dis  # torch Tensor
        }
    

    def _cal_rewards(self) -> Union[torch.FloatType, torch.Tensor]:
        if self.poses is None: 
            default_envs = torch.arange(self.n_envs, device=gs.device, dtype=torch.int32)
            self._get_poses(default_envs)
            
        cur_pos = self.poses['cur_Tpose']
        tar_pos = self.poses['target_Tpose']

        dis = torch.linalg.norm(cur_pos[:, :2] - tar_pos[:, :2])
        ang = torch.abs(cur_pos[:, 5] - tar_pos[:, 5])

        return 1 / (dis/self.ini_delta_dis + 0.1) / torch.sqrt(ang/self.ini_delta_ang + 0.1) - self.reward0


    def _ikine(self, link: gs.engine.entities.rigid_entity.RigidLink, 
               pos: torch.Tensor, quat: torch.Tensor, 
               envs_idx: torch.Tensor) -> torch.Tensor: 
        qpos = self.robot.inverse_kinematics(
            link=link,
            init_qpos = self.robot.get_dofs_position(envs_idx = envs_idx),
            pos=pos,  # torch tensor
            quat=quat,  # torch tensor  
            dofs_idx_local=self.robot_dofs_idx[0:7],
            envs_idx=envs_idx  # torch tensor
        )
        return qpos
    

    def _cal_intersection(self) -> List[float]:
        if self.keypoints is None:
            return []
            
        cur_points_np = to_numpy(self.keypoints['cur_keypoints'][:, :, :2])
        target_points_np = to_numpy(self.keypoints['target_keypoints'][:, :, :2])
        print(cur_points_np)
        print(target_points_np)
        ratio: List[float] = []
        for i in range(cur_points_np.shape[0]):
            try:
                # Check for invalid values
                cur_points = cur_points_np[i]
                tar_points = target_points_np[i]

                if not np.all(np.isfinite(cur_points)) or not np.all(np.isfinite(tar_points)):
                    print(f"Env {i}: Invalid values detected")
                    ratio.append(0.0)
                    continue

                # Check for duplicate/collinear points
                unique_cur = np.unique(cur_points, axis=0)
                unique_tar = np.unique(tar_points, axis=0)

                if len(unique_cur) < 3 or len(unique_tar) < 3:
                    print(f"Env {i}: Not enough unique points")
                    ratio.append(0.0)
                    continue

                cur_polygon = Polygon(cur_points)
                tar_polygon = Polygon(tar_points)

                if not cur_polygon.is_valid or not tar_polygon.is_valid:
                    print(f"Env {i}: Invalid polygon")
                    ratio.append(0.0)
                    continue

                intersection_geom = cur_polygon.intersection(tar_polygon)
                area = intersection_geom.area
                ratio.append(area / cur_polygon.area if cur_polygon.area > 0 else 0.0)

            except Exception as e:
                print(f"Env {i}: Polygon error - {e}")
                ratio.append(0.0)
        print(ratio)
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
        """
        Args:
            envs_idx: torch.Tensor
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
              env_separate=True, seed=[0])
    env.seed([5004])
    env.reset()

    env.start_recording()
    for i in range(500):
        env.step(action=torch.tensor([[0.3, 0.3, 0.3]]))
        if i % 10 == 0: 
            # env.reset()
            # print(env.calculate_all_keypoints())
            print(np.shape(env.render()[0]))
            # print(env._cal_intersection())
            # print(env.get_key_points())

    env.stop_recording()