import numpy as np

class SpawnCircularSampler:
    def __init__(self, 
                 radius_inner:float, 
                 radius_outer:float, 
                 angle_min:float, 
                 angle_max:float, 
                 min_central_sep=None, 
                 sample_slackness = 4, # used for controlling the scale of normal distribution for the radius sampling
                 ):
        
        self.radius_inner = radius_inner
        self.radius_outer = radius_outer
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.min_central_sep = min_central_sep
        self.sample_slackness = sample_slackness

    def sample(self, sample_counts, sample_generator=None,
               max_retries = 64):
        rng = sample_generator if sample_generator is not None else np.random.default_rng()

        spawn_samples = {
            'block_pos': [],
            'block_angle': [],
            'target_pos': [],
            'target_angle': [],
        }
        for _ in range(sample_counts):
            idx = 0
            while(idx<max_retries):
                radius_sampled = np.clip(rng.normal(loc=((self.radius_inner + self.radius_outer) / 2),
                                                  scale=((self.radius_outer - self.radius_inner) / self.sample_slackness),
                                                  size=(2 )),
                                                  a_max=self.radius_outer,
                                                  a_min=self.radius_inner)

                angle_sampled = rng.uniform(low=self.angle_min,
                                                  high=self.angle_max,
                                                  size=(2 ))
                print(radius_sampled)
                pos = self._polar2xy(radius=radius_sampled, angle=angle_sampled)
                block_pos:np.array = pos[0]
                target_pos:np.array = pos[1]

                if self._is_sep(block_pos, target_pos):
                   break
                idx = idx + 1

            spawn_samples['block_pos'].append(block_pos)
            spawn_samples['block_angle'].append(angle_sampled[0])
            spawn_samples['target_pos'].append(target_pos)
            spawn_samples['target_angle'].append(angle_sampled[1])
        return spawn_samples

    def _polar2xy(self, radius, angle):
        return np.array([radius * np.cos(angle), radius * np.sin(angle)]).swapaxes(0, 1)

    def _is_sep(self, block_pos, target_pos):
        if self.min_central_sep is not None:
            return np.linalg.norm(block_pos - target_pos) > self.min_central_sep
        else: return True


if __name__ == "__main__":
    SpawnCircularSamplercls = SpawnCircularSampler(radius_inner=0.3, 
                                                radius_outer=0.5, 
                                                angle_min=np.pi/2, 
                                                angle_max=np.pi, 
                                                min_central_sep=0.1, 
                                                sample_slackness=4)
    SpawnCircularSamplercls.sample(10, None, 30)


