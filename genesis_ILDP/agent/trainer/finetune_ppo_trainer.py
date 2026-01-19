import torch 
from genesis_ILDP.agent.trainer.base_trainer import BaseTrainer
from genesis_ILDP.agent.rl_agent.base_agent import BaseAgent

MAX_LOOP_ITERATION = 100000
class finetune_ppo_trainer(BaseTrainer):
    def __init__(self, ft_diff_steps, gamma, gae_lambda, 
                 agent:BaseAgent, env_collector, evaluator, batch_size=64,
                   train_epochs=10, per_epoch_iteration=3):
        super().__init__(ft_diff_steps=ft_diff_steps, gamma=gamma, gae_lambda=gae_lambda,
                         agent=agent, env_collector=env_collector, evaluator=evaluator, batch_size=batch_size, 
                         train_epochs=train_epochs, per_epoch_iterations=per_epoch_iteration)

        
    def train_step(self, ):
        """
        train_step: run one training epoch on one time for collected data from env_collector during the training stage
        """

        iteration_idx = 0
        while(iteration_idx < self.per_epoch_iteration): 
            iteration_idx = iteration_idx + 1
            safe_idx = 0
            while(safe_idx < MAX_LOOP_ITERATION):
                imgs, agent_pos, chain_before, chain_after, advantage, oldvalue, oldlogprob, denoising_idx = self._sample_train_batch_from_buffer(self.batch_size)
                if imgs is None: break
                obs = {
                    'imgs': imgs, 
                    'agent_pos': agent_pos
                }
            
                self.agent.train_step(chain_before, chain_after, advantage, obs, denoising_idx, oldvalue, oldlogprob)

    
    def _sample_train_batch_from_buffer(self, batch_size):

        """
        _sample_train_batch_from_buffer: sample from train_batch saved into self.buffer
        """
        if self.idx_pointer == len(self.global_idx): return tuple([None] * 8)

        cur_global_idx = self.global_idx[self.idx_pointer,]
        cur_local_idx = self.local_idx[self.idx_pointer, self.idx_pointer + batch_size]
        # prepare training samples from buffers
        imgs = self.buffer['obs']['imgs'][cur_global_idx, cur_local_idx]
        agent_pos = self.buffer['obs']['agent_pos'][cur_global_idx, cur_local_idx]
        chain_before = self.buffer['action'][cur_global_idx, cur_local_idx + 1]
        chain_after = self.buffer['action'][cur_global_idx, cur_local_idx] # smaller idx means denoised more steps? confirmation 
        advantage = self.buffer['advantage'][cur_global_idx] # advantage is being regarded as gae in ppo
        old_value = self.buffer['oldvalue'][cur_global_idx]
        oldlogprob = self.buffer['oldlogprob'][cur_global_idx, cur_local_idx]

        # cur_local_idx = denoising idx from collected datasets (starting from 0 - self.diff_steps - 1)
        return imgs, agent_pos, chain_before, chain_after, advantage, old_value, oldlogprob, cur_local_idx
    

    def _get_train_data(self) -> None:
        collect_trajs = self.env_collector.collect_trajs
        collected_steps = collect_trajs['length']

        total_indices = self.buffer_size * self.ft_diff_steps
        self.order = torch.permute(total_indices, device=self.device)
        shape = (collected_steps, self.ft_diff_steps)
        self.global_idx, self.local_idx = torch.unravel_index(self.order, shape=shape, )
        self.idx_pointer = 0

        # save data into buffer inside the trainer for future usage 
        self.buffer['obs']['imgs'] = collect_trajs['obs']['imgs'] # shape for imgs and agent_pos should include dim of horizon
        self.buffer['obs']['agent_pos'] = collect_trajs['obs']['agent_pos']
        self.buffer['chain_before'] = collect_trajs['chain_before']
        self.buffer['chain_after'] = collect_trajs['chain_after']
        self.buffer['reward'] = collect_trajs['reward']
        self.buffer['episode_ends'] = collect_trajs['episode_ends']
        self.buffer['truncate'] = collect_trajs['truncate']
        self.buffer['logprob'] = collect_trajs['logprob']
        # evaluate the old value once using the self.agent.critic model for ppo model for reference
        self.buffer['oldvalue'] = self.agent.critic(imgs=self.buffer['obs']['imgs'], agent_pos=self.buffer['obs']['agent_pos'])
        # calculate the total advantages for the full trajs (independentely from each trajs, maintained inside rl_utils.py)
        # it should use truncate, reward and ...needed information for calculating the advantage
        # TODO