from genesis_ILDP.policy.action_diffusion_image_policy import ActionDiffusionImagePolicy
from matplotlib import pyplot as plt
import numpy as np
from genesis_ILDP.utils.cuda import to_numpy

shape_meta = {
    "action": {
       'shape': (2, )
    },
    "obs":{
        "image": {
            'shape': (3, 96, 96)
        },
        
        "agent_pos":{
            'shape': (2, )
        } 
    }
}
diff_steps = 100
policy = ActionDiffusionImagePolicy(shape_meta=shape_meta, diff_steps=diff_steps)

betas_linear, cum_alphas_linear = policy.NoiseScheduler(mode='Linear')
betas_root_linear, cum_alphas_root_linear = policy.NoiseScheduler(mode='RootLinear')
betas_cosine, cum_alphas_cosine = policy.NoiseScheduler(mode='Cosine')
print("beta inside Cosine scheduler", betas_cosine)
print("cum_alphas inside Cosine scheduler", cum_alphas_cosine)

idx = diff_steps - 1 
idx = 1
variance = betas_cosine[idx] * (1 - cum_alphas_cosine[idx-1]) / (1 - cum_alphas_cosine[idx])
# variance = betas_linear[idx] * (1 - cum_alphas_linear[idx-1]) / (1 - cum_alphas_linear[idx])
print(f"variance added to the [idx={idx}] diffusion step(idx = 1 means the last step)", variance)
# print("Cum alphas from the Root linear scheduler: ", cum_alphas_root_linear[-1])
# print(cum_alphas_root_linear.dtype)
plt.figure(figsize=(8, 8))

counts = betas_cosine.shape[0]
idx = np.linspace(0, counts-1, counts)

plt.plot(idx, to_numpy(cum_alphas_linear), label='linear')
plt.plot(idx, to_numpy(cum_alphas_cosine), label='cosine')
plt.plot(idx, to_numpy(cum_alphas_root_linear), label='root_linear')

plt.legend(loc='best')
plt.show()
