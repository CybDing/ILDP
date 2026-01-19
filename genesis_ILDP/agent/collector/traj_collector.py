import torch
import numpy as np
from genesis_ILDP.workspace.base_workspace import BaseWorkspace
from genesis_ILDP.workspace.train_action_diffusion_image_workspace import TrainActionDiffusionImageWorkspace
from genesis_ILDP.assets.backup.eval_action_diffusion_image_workspace import EvalActionDiffusionImageWorkspace
import copy
import hydra

class TrajCollector:

    def __init__(self, eval_cfg, seed=50000, n_episodes=200, max_step=500):
        self.workspace = EvalActionDiffusionImageWorkspace \
                        (eval_cfg, output_dir=None, enable_render=False)
        self.policy = None
        self.env_runner = None
        self.seed = seed # the initial seed which the collector is gonna to collect for the first time. 

        # self.workspace.cfg is the checkpoint cfg
        checkpoint_cfg = copy.deepcopy(self.workspace.cfg)

        from genesis_ILDP.dataset.base_dataset import BaseImageDataset
        dataset: BaseImageDataset = hydra.utils.instantiate(checkpoint_cfg.task.dataset)
        normalizer = dataset.get_normalizer()

        self.workspace.model.set_normalizer(normalizer)

        if self.policy is None: 
            self.policy = self.workspace.model
        
        if self.env_runner is None:
            checkpoint_cfg.env_runner.n_train = 0
            checkpoint_cfg.env_runner.n_test = n_episodes
            checkpoint_cfg.env_runner.n_train_vis = 0
            checkpoint_cfg.env_runner.n_test_vis = n_episodes

            checkpoint_cfg.env_runner.test_start_seed = self.seed
            checkpoint_cfg.env_runner.max_step = max_step

            # Uss the same settings for generation of targets and object positions from the pretrained version 
            self.env_runner = hydra.utils.instantiate(checkpoint_cfg.env_runner)
   
    def collect_trajs(self, seed=None):
        if seed is not None:
            self.env_runner._update_seeds(seed_train=None, seed_test=seed) # only use one field of seed for runner when collecting data for finetuning
        _ = self.env_runner.run() 
        collected_trajs = self.env_runner.get_saved_trajectories() # get the episode_buffers in dict type which includes 
        # n_episode episodes collected via test_seed + k * n_episode

        return collected_trajs

    def load_state_dict(self, state_dict):
        '''
        load_state_dict: load new diffusion policy state_dict(e.g. 'action_diffusion_image_policy' class state_dict)
        
        :param state_dict: the new policy state_dict passed when the reinforcement learning algorithm is online, and needs to collect NEW buffers
        '''
        self.policy.load_state_dict(state_dict)

    
        