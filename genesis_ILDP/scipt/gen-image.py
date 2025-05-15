import PIL.Image
import torch
import numpy
import PIL
import os

from genesis_ILDP.env.pushT_env import PushTEnv

size = 256
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), './img')
os.makedirs(path, exist_ok=True)  # 确保目录存在

env = PushTEnv(render_size=size)
env.start(show_camera=False)


for i in range(10):
    img = env.render('rgb_array')[0]
    image = PIL.Image.fromarray(img.astype('uint8'), 'RGB')
    img_name = f'PushTEnv-{i}.jpg'
    image.save(os.path.join(path, img_name))
    env.reset()
    




