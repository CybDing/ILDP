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
    gae_advantage = torch.zeros_like(reward_trajs)
    temp_value = 0
    for i in reversed(range(len(reward_trajs))):
        temp_value = delta[i] + gamma * gae_lambda * is_truncate[i] * temp_value
        gae_advantage[i] = temp_value

    return gae_advantage


def calculate_full_batch_gae(value_trajs, last_obs_value,
                             reward_trajs, is_truncate_trajs, episode_ends_trajs,
                             gamma, gae_lambda) -> torch.Tensor:

    idx = torch.where(episode_ends_trajs == 1)[0]
    idx = torch.cat([torch.tensor([0], device=idx.device, dtype=idx.dtype), idx], dim=0)
    final_gae = list()
    for i in range(len(idx) - 1):
        ep_start_idx = idx[i]
        ep_end_idx = idx[i+1] + 1

        full_value_trajs = torch.cat([value_trajs[ep_start_idx:ep_end_idx], last_obs_value[i].unsqueeze(0)], dim=0)
        final_gae.append(calculate_GAE(reward_trajs=reward_trajs[ep_start_idx:ep_end_idx],
                                       value_trajs=full_value_trajs,
                                       is_truncate=is_truncate_trajs[ep_start_idx:ep_end_idx],
                                       gamma=gamma, gae_lambda=gae_lambda))

    return torch.cat(final_gae, dim=0)
