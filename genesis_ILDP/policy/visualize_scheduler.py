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
policy = ActionDiffusionImagePolicy(shape_meta=shape_meta)

_, betas_scheduler_linear = policy.NoiseScheduler(mode='Linear')
_, betas_scheduler_root_linear = policy.NoiseScheduler(mode='RootLinear')
_, betas_scheduler_cosine = policy.NoiseScheduler(mode='Cosine')

plt.figure(figsize=(8, 8))

counts = betas_scheduler_cosine.shape[0]
idx = np.linspace(0, counts-1, counts)

plt.plot(idx, to_numpy(betas_scheduler_linear), label='linear')
plt.plot(idx, to_numpy(betas_scheduler_cosine), label='cosine')
plt.plot(idx, to_numpy(betas_scheduler_root_linear), label='root_linear')

plt.legend(loc='best')
plt.show()
