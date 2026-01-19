import torch 

def calculate_GAE(reward_trajs, value_trajs, is_truncate, gamma, gae_lambda):
    """
    Docstring for calculate_GAE: calculate the general advantage estimation for collected replay buffer. 
    
    :param reward_trajs: (N, )
    :param value_trajs: (N, ) use default current Q net for calculating the current obs trajs' values(used for bootstrapping)
    # :param episode_ends: (N, ) if this idx is the last step in one episode, the episode_end[idx] = 1
    :param terminated: (N, )
    """
    
    delta = reward_trajs + gamma * value_trajs[1:] * is_truncate - value_trajs[:-1]
    gae_advantage = list()
    temp_value = 0
    for i in range(len(reward_trajs)):
        temp_value = temp_value + (gae_lambda ** i)* delta[-i]
        gae_advantage[i] = temp_value
    
    gae_advantage[1::] = gae_advantage[-1:-1:]

    return gae_advantage


def calculate_full_batch_gae(self, value_trajs, last_obs_value, 
                                 reward_trajs, is_truncate_trajs, episode_ends_trajs)-> torch.tensor:
       
        idx = torch.where(episode_ends_trajs == 1)[0]
        idx = torch.cat([torch.tensor([0], device=idx.device, dtype=idx.dtype), idx], dim=0)
        final_gae = list()
        for i in range(len(idx) - 1):
           ep_start_idx = episode_ends_trajs[i]
           ep_end_idx = episode_ends_trajs[i+1] + 1 # since the episode_ends save for the last step

           full_value_trajs = torch.cat([value_trajs[idx[ep_start_idx:ep_end_idx]], last_obs_value[i].unsqueeze(0)], dim=0)
           final_gae.append(calculate_GAE(value_trajs=full_value_trajs, rewards_trajs=reward_trajs[ep_start_idx:ep_end_idx],
                                           is_truncate=is_truncate_trajs[ep_start_idx:ep_end_idx], 
                                           gamma=self.gamma, gae_lambda=self.gae_lambda)) 

        return final_gae 
