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
file_path = os.path.dirname(os.path.abspath(__file__)) # Return the directory path for current file
urdf_path = os.path.join(file_path, '../assets/roboticArms/urdf/rizon44/flexiv_rizon4_kinematics.urdf')

class PushTEnv():
    # TODO reset(), step(), render(), close(), seed(), run(), cal_planner()
    # TODO _get_obs(), get_info()
    metadata = {"render.mode": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self,
                 render_size=96,
                 xlim=2.,
                 ylim=2.,
                 seed=None, # seed 
                 urdf_path=urdf_path,
                 cube_w = 0.1
                 ):
        # super().__init__()

        self.sim_hz = 100
        self.control_hz = 50
        self.is_init = False
        self._seed = seed
        self.scene = None
        self.n_envs = None
        self.np_random = None
        self.block_lim = {'xlim': xlim/2, 'ylim': ylim/2}
        self.cube_w = cube_w

        self.observation_space = spaces.Dict({
            'images': spaces.Box(
                low=0.,
                high=1.,
                shape=(3, render_size, render_size),
                dtype = np.float64
            ),
            'agent_pos': spaces.Box(
                low=np.array([-xlim, -ylim], dtype=np.float32),
                high=np.array([-xlim, -ylim], dtype=np.float32),
                shape=(2, ),
                dtype=np.float32
            )
        })
        self.action_space = spaces.Box(
            low=np.array([-xlim, -ylim], dtype=np.float32),
            high=np.array([-xlim, -ylim], dtype=np.float32),
            shape=(2, ),
            dtype=np.float32
        )

        self.seed()


    def start(self, n_envs=1, show_interact_viewer=False, show_camera=False):
        # init only once
        assert self.is_init == False 
        self.n_envs = n_envs
        gs.init(
            seed = self._seed,
            backend = gs.gpu
        )
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=1/self.sim_hz, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 * self.sim_hz),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40 # angle look at
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),

            rigid_options=gs.options.RigidOptions(
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_interact_viewer,
        )
        self.plane = self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True)) # add a plane(with special markers)

        self.robot : gs.engine.entities.RigidEntity = self.scene.add_entity(
            gs.morphs.URDF(
                pos = (0, 0, 0.5),
                file = urdf_path,
                fixed=True,
                collision=True,
                links_to_keep=[
                    'tcp',
                    # 'grav_base_link',
                ]
            )
        )
        # USE Box(could not identify mass?)
        self.cube : gs.engine.RigidEntity = self.scene.add_entity(
            gs.morphs.Box(
                pos=(1, 1, self.cube_w),
                size = (self.cube_w, self.cube_w, self.cube_w)
            )
        )
        self.cam = self.scene.add_camera(
            res=(120, 120),
            pos=(0,0,2),
            lookat=(0,0,0),
            fov=40,
            GUI=show_camera
        )
        self.scene.build(n_envs=n_envs)

        jnt_names = [ 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 
                      'joint6', 'joint7', 'finger_width_joint'] 
        self.dofs_idx = [self.robot.get_joint(name).dof_idx_local for name in jnt_names]
        self.eef: gs.engine.entities.rigid_entity.RigidLink = self.robot.get_link('tcp')
        self.eef_idx = self.eef.idx_local

        self.robot.set_dofs_kp(
            kp             = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100]),
            dofs_idx_local = self.dofs_idx,
        )
        self.robot.set_dofs_kv(
            kv             = np.array([450, 450, 350, 350, 200, 200, 200, 10]),
            dofs_idx_local = self.dofs_idx,
        )
        self.render_cache = None
        self.reset()

    # setting seed for generator
    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)


    def reset_idx(self, envs_idx : list):
        num_reset = len(envs_idx)
        if num_reset == 0: return

        block_pos = to_torch(np.concatenate(
                    (self.np_random.random(size=(num_reset, 1)) * self.block_lim['xlim'] * 2 - self.block_lim['xlim'],
                     self.np_random.random(size=(num_reset, 1)) * self.block_lim['ylim'] * 2 - self.block_lim['ylim'],
                     np.ones(shape=(num_reset,1)) * self.cube_w / 2
                     ), axis=1))
        block_angle = self.np_random.random(size=(num_reset, )) * np.pi / 2 # 0 - pi/2 

        target_pos = to_torch(np.concatenate(
                    (self.np_random.random(size=(num_reset, 1)) * self.block_lim['xlim'] * 2 - self.block_lim['xlim'],
                     self.np_random.random(size=(num_reset, 1)) * self.block_lim['ylim'] * 2 - self.block_lim['ylim'],
                     np.ones(shape=(num_reset,1)) * self.cube_w / 2
                     ), axis=1))
        target_angle = self.np_random.random(size=(num_reset, )) * np.pi / 2 

        home_pos = torch.zeros(size=(num_reset, len(self.dofs_idx)))

        self.robot.set_dofs_position(home_pos, 
                                     dofs_idx_local= self.dofs_idx, 
                                     envs_idx=envs_idx,
                                     zero_velocity=True)
        self.cube.set_pos(block_pos, envs_idx)


    def step(self, action):
        # action: agent_pos(eef_pos)
        self.scene.step()
    

    def ikine(self, ): # 简单测试是否需要重写
        pass


    def reset(self,):
        self.reset_idx(envs_idx=[i for i in range(self.n_envs)])

    def _get_obs(self, depth=False, segmentation=False, normal=False):
        img = self.cam.render(depth=depth, segmentation=segmentation, normal=normal) # img <tuple> (res[0], res[1], 3)
        self.render_cache = img

        # jnt_pos = [self.robot.get_dofs_position(idx, envs_idx=np.arange(self.n_envs)) for idx in self.dofs_idx]
        agent_pos = self.robot.get_links_pos(self.eef_idx, envs_idx=np.arange(self.n_envs))
        obs = {
            'image': img,
            'agent_pos': agent_pos
        }
        return obs
    
    def render(self, mode):
        assert mode == 'rgb_array'
        self._get_obs()
        # if self.render_cache is None:
        #     self._get_obs()
        return self.render_cache
    

# test: script python -m genesis-ILDP.env.pushT_env
if __name__ == '__main__':
    env = PushTEnv()
    env.start(show_camera=False)

    for i in range(1000):
        env.step()
        if i % 10 == 0: # render per 10 frames
            env.render('rgb_array')
    