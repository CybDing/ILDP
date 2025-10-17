from diffusion_policy.diffusion_policy.model.vision.crop_randomizer import CropRandomizer
from genesis_ILDP.dataset.pusht_image_dataset import PushTImageDataset
import matplotlib.pyplot as plt
img_size = (3, 96, 96)
crop_sz = (86, 86)
cropper = CropRandomizer(input_shape=img_size, crop_height = crop_sz[0], crop_width=crop_sz[1], num_crops=8, pos_enc=False)

cropper.train()

zarr_path = '../data/train_data/pusht/merged_data_0925.zarr'     
dataset = PushTImageDataset(zarr_path, horizon=16)

test_img = dataset.__getitem__(30)['obs']['image'][0].unsqueeze(dim=0)
# print(test_img.shape)
img_batched = cropper(test_img)

plt.figure(figsize=(8, 4))

for i in range(8):

    plt.subplot(2, 4, i+1)
    plt.imshow(img_batched[i].permute(1, 2, 0).cpu().numpy())

plt.show()