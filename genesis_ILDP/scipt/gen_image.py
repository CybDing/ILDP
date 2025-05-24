import PIL.Image
import torch
import numpy as np
import PIL
import os

from genesis_ILDP.env.pushT_env import *

size = 256

def count_unique_link(arr: np.array):
    unique_nonzero = np.unique(arr[arr != 0])
    return unique_nonzero, len(unique_nonzero)

def img_process(arr: np.array, ind):
    arr = np.where(arr != 0 and arr != ind, 0.5 * 255, arr)
    return np.where(arr == ind, 255, arr)

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/img')
os.makedirs(path, exist_ok=True) 

raw_path = os.path.join(path, 'raw')
seg_path = os.path.join(path, 'seg')

os.makedirs(raw_path, exist_ok=True)
os.makedirs(seg_path, exist_ok=True)

env = PushTEnv(render_size=size)
env.start(show_camera=False)

for i in range(10):

    for j in range(10): env.step()
    img = list(env.render('rgb_array'))

    ind, _ = count_unique_link(img[2])
    print(ind)
    # index ?
    img[2] = img_process(img[2], 19)

    image_rgb = PIL.Image.fromarray(img[0].astype('uint8'), 'RGB')
    image_L = PIL.Image.fromarray(img[2].astype('uint8'), 'L')

    img_name = f'PushTEnv-{i}.jpg'  
    raw_prefix = 'raw-'
    seg_prefix = 'seg-'
    image_rgb.save(os.path.join(raw_path, raw_prefix, img_name))
    image_L.save(os.path.join(seg_path, seg_prefix, img_name))

    env.reset()