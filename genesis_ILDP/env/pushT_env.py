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

from genesis_ILDP.utils.cuda import *
from genesis_ILDP.config.env_config import *

class PushTEnv(gym.Env):
    # TODO reset(), step(), render(), close(), seed()
    # TODO get_info()
    metadata = {"render.mode": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self,
                 render_size=128,
                 xlim=.5,
                 ylim=.5,
                 seed=None, # seed 
                 path=env_path,
                #  cube_w = 0.1
                 ):
        # super().__init__()

        self.render_size = render_size
        self.sim_hz = 200
        self.control_hz = 50 # how long waiting for robotic arms to finsih exectuing an action
        self.is_init = False
        self._seed = seed
        self.scene = None
        self.n_envs = None
        self.np_random = None
        self.block_lim = {'xlim': xlim, 'ylim': ylim}
        # self.cube_w = cube_w
        self.path = path
        self.env_seed = None 

        self.observation_space = spaces.Dict({
            'images': spaces.Box(
                low=0.,
                high=1.,
                shape=(3, render_size, render_size),
                dtype = np.float64
            ),
            'agent_pos': spaces.Box(
                low=np.array([-xlim, -ylim], dtype=np.float32),
                high=np.array([xlim, ylim], dtype=np.float32),
                shape=(2, ),
                dtype=np.float32
            )
        })
        self.action_space = spaces.Box(
            low=np.array([-xlim, -ylim], dtype=np.float32),
            high=np.array([xlim, ylim], dtype=np.float32),
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
            sim_options=gs.options.SimOptions(dt=1/self.sim_hz, substeps=1),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(self.sim_hz),
                camera_pos=(2.5, 0.0, 2.5),
                camera_lookat=(-0.12, -0.12, 0.6),
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
                dt=1/self.sim_hz
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
                pos = (0, 0, 0.5),
                file = self.path['robot'],
                fixed=True,
                collision=True,
                links_to_keep=[
                    'tcp',
                    # 'grav_base_link',
                ]
            )
        )

        self.cube : gs.engine.entities.RigidEntity = self.scene.add_entity(
            gs.morphs.URDF(
                pos = (0, 0, 0),
                file=self.path['cube'],
                collision=True,
                fixed=True,
                visualization=True,
                links_to_keep=[
                    'cubee',
                ]
                )
        )

        # self.cube : gs.engine.RigidEntity = self.scene.add_entity(
        #     gs.morphs.Box(
        #         pos=(0.5, 0.5, 0.1),
        #         size = (0.1, 0.1, 0.1)
        #     )
        # )
        # box_baselink_joint, box_baselink

        self.cam = self.scene.add_camera(
            res=(self.render_size, self.render_size),
            pos=(0.25, 1, 2),
            lookat=(0.25, 0, 0.0),
            fov=40,
            GUI=show_camera,
        )
        self.scene.build(n_envs=n_envs)
        # print(self.cube.get_joint('cube_plane_joint').dof_idx_local)

        # print(self.cube)
        jnt_names = [ 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 
                      'joint6', 'joint7', 'finger_width_joint'] 

        self.robot_dofs_idx = [self.robot.get_joint(name).dof_idx_local for name in jnt_names]
        self.cube_dofs_idx  = self.cube.get_joint('cube_plane_joint').dof_idx_local
        self.tcp: gs.engine.entities.rigid_entity.RigidLink = self.robot.get_link('tcp')
        self.tcp_idx = self.tcp.idx_local
        self.marker_idx = self.plane.get_link('marker').idx_local
        self.marker_dofs_idx = self.plane.get_joint('marker_joint').dof_idx_local
        self.robot.set_dofs_kp(
            kp             = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100]),
            dofs_idx_local = self.robot_dofs_idx,
        )
        self.robot.set_dofs_kv(
            kv             = np.array([450, 450, 350, 350, 200, 200, 200, 10]),
            dofs_idx_local = self.robot_dofs_idx,
        )
        self.render_cache = None
        # self.seed(seed=seed)
        # self.reset()
        self.seed()

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
            
            block_x = rng.random() * self.block_lim['xlim']
            block_y = rng.random() * self.block_lim['ylim'] * 2 - self.block_lim['ylim']
            block_z = 0.05  # 固定高度
            
            block_angle = rng.random() * np.pi / 2
            
            target_x = rng.random() * self.block_lim['xlim']
            target_y = rng.random() * self.block_lim['ylim'] * 2 - self.block_lim['ylim']
            target_z = 0.0  # 目标在地面
            
            target_angle = rng.random() * np.pi - np.pi / 2
        
            block_positions.append([block_x, block_y, block_z])
            target_positions.append([target_x, target_y, target_z])
            block_angles.append(block_angle)
            target_angles.append(target_angle)
    
        block_pos = np.array(block_positions)
        target_pos = np.array(target_positions)
        block_angle = np.array(block_angles).reshape(-1, 1)
        target_angle = np.array(target_angles).reshape(-1, 1)
        
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
        
        home_pos = torch.zeros(size=(num_reset, len(self.robot_dofs_idx)), device=gs.device)

        self.robot.set_dofs_position(
            position=home_pos,
            dofs_idx_local=self.robot_dofs_idx,
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
        
        observation = self._get_obs(rgb=True, envs_idx=envs_idx)
        return observation

    def reset(self,):
        self.reset_idx(envs_idx=torch.arange(self.n_envs, device=gs.device, dtype=torch.int16)
    )
        
    def step(self, action=None, envs_idx = None):
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
               quat = torch.tile(torch.tensor([0, 0, 1, 0]), (len(envs_idx), 1))
               qpos = self._ikine(self.tcp, action, quat, envs_idx)
               self.robot.control_dofs_position(position=qpos[:, 0:7], # does not control tcp joints
                                        dofs_idx_local=self.robot_dofs_idx[0:7], 
                                        envs_idx=envs_idx
                                     ) 
        for _ in range(n_steps):
            self.scene.step()

        # TODO done condition
        observation = self._get_obs(rgb=True, envs_idx=envs_idx)
        info = self._get_info(envs_idx=envs_idx)
        done = False
        reward = 0

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
        # CHECK CWD -> filepath 
        # target_folder = "./ILDP/genesis_ILDP/test/"  
        os.makedirs(target_folder, exist_ok=True)
        old_dir = os.getcwd()
        os.chdir(target_folder)

        if filename is None:
            filename = os.path.join(target_folder, time.strftime("%Y%m%d-%H-%M") + "-pushT-env.mp4")
        self.cam.stop_recording(save_to_filename=filename)
        
        os.chdir(old_dir)

    def _ikine(self, link, pos, quat, envs_idx): 
        qpos = self.robot.inverse_kinematics(
        link = link,
        pos = pos,
        quat = quat,
        dofs_idx_local=self.robot_dofs_idx[0:7],
        envs_idx=envs_idx
       )
        return qpos


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
        agent_pos = self.robot.get_links_pos(self.tcp_idx, envs_idx)
        idx = to_numpy(envs_idx, float=False)
        obs = {
            'envs_idx': envs_idx,
            'image': img[0][idx,:],
            'agent_pos': agent_pos[idx, :]
        }
        marker_pos = self.plane.get_links_pos(self.marker_idx, envs_idx)    
       #  print(marker_pos)    
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
            "agent_pos": self.tcp.get_pos()[idx, :],
            "goal_pos": self.plane.get_links_pos(self.marker_idx)[idx, :],
        }
        return info
    
    
if __name__ == '__main__':
    env = PushTEnv()
    env.start(n_envs=10, show_camera=False, show_interact_viewer=False, 
              env_separate=True, seed=[0, 1])
    env.seed(np.arange(10))
    env.reset()

    # env.start_recording()
    for i in range(500):
        # env.step(action=torch.tensor([[0.1, 0.1],
        #                               [0.1, 0.1]]))
        env.step()
        env._get_obs()
        if i % 250 == 0: 
            env.reset()
            print(np.shape(env.render()[0]))
            # Problem: Changing camera rendering modes(for each arm)
        print(np.shape(env.render()[0])) 
    # env.stop_recording()