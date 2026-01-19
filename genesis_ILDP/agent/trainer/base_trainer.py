import torch
import torch.nn
from genesis_ILDP.model.rl.rl_utils import calculate_GAE
from genesis_ILDP.agent.collector.traj_collector import TrajCollector
class BaseTrainer:
    def __init__(self, 
                  train_epochs, # finetuning epochs policy using rl_agent(each epoch will collect some new data)
                  per_epoch_iterations, # how many times we train on the data we collected one time from the the env collector 
                  batch_size, # how many samples does each training step inside the training iteration would use for calulatingt the overall gradient
                  ft_diff_steps, gamma, gae_lambda, 
                  agent,
                  env_collector:TrajCollector,
                  evaluator, ):
        self.ft_diff_steps = ft_diff_steps
        self.buffer = None # replay buffer saving
        self.buffer_size = None
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.agent = agent

        # buffer saving for shuffling collected trajs from env_collector
        self.order = None 
        self.global_idx = None
        self.local_idx = None



    def step_trainer(self, ):
        """
        step_trainer: train one step by first collecting data using old policy, then save the data into buffer,
        and train using small batch using ppo_diffusion 'loss()' method for policy improvement. 
        
        :param 
        """

    
    def _get_train_data(self, ): 
        """
        _get_train_data: return data organized in style like obs, reward, action
        """
        # for online reinforcement learning, then we should collect the trajs every time we need them, 
        # for offline reinforcement learning, the replay buffer could be rotated buffer saving all the data inside
        raise NotImplementedError

    def _sample_train_batch_from_buffer(self, batch_size, *args):
        raise NotImplementedError



    def _log_wandb(self, info):
        """
        Docstring for _log_wandb: use logger for uploading wandb information online, and make training visualize correctly
        
        :param self: Description
        """