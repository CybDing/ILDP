### Copied From diffusion policy repository 

from typing import Dict
# from genesis_ILDP.policy.base_image_policy import BaseImagePolicy

class BaseImageRunner:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def run(self, policy) -> Dict:
        raise NotImplementedError()
