from PIL import Image
import matplotlib.pyplot as plt
import config
import numpy as np
import config
from torch import nn
from torch import optim
from utils import load_checkpoint
from model import HAN
from glob import glob
import albumentations as A
import torch
import pathlib
import cv2
from torchvision import transforms
import augmentations
import math

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def recreate_image(image, gen, scale_factor):
    original_image = image
    patch_size_lr = config.LOW_RES
    channels, width, height = image.shape
    low_width = width//scale_factor
    low_height = height//scale_factor
    hr_image = torch.empty_like(image)
    image = transforms.Resize((low_width, low_height))(image)
    # image = augmentations.highres_transform(image=image)["image"]
    # image = torch.from_numpy(image).permute(2, 0, 1)
    nx_patches = int(low_width*1.1/patch_size_lr)+1
    ny_patches = int(low_height*1.1/patch_size_lr)+1
    x_steps = np.linspace(0, low_width, nx_patches,
                          dtype=np.int, endpoint=False)
    y_steps = np.linspace(0, low_height, ny_patches,
                          dtype=np.int, endpoint=False)
    x_steps[-1] = low_width-patch_size_lr
    y_steps[-1] = low_height-patch_size_lr

    # Uses batches to call the generator fewer times
    n_patches = nx_patches*ny_patches
    batch_size = 16
    pos = [(x, y) for x in x_steps for y in y_steps]
    n_batches = int(np.ceil(n_patches/batch_size))
    last_batch = n_patches % batch_size
    batches = [pos[x*batch_size:x*batch_size+batch_size]
               for x in range(n_batches)]

    for b in batches:
        n_patches = len(b)
        lr_patch = torch.empty(
            (n_patches, channels, patch_size_lr, patch_size_lr))
        for i in range(n_patches):
            x = b[i][0]
            y = b[i][1]
            lr_patch[i, :, :, :] = image[:, x:x +
                                         patch_size_lr, y:y+patch_size_lr]
        lr_patch = lr_patch.cuda()
        hr_patches = gen(lr_patch)  # .permute(0, 2, 3, 1)
        hr_patches = hr_patches.cpu().detach()
        # The edgeds of the image don't look as good as the center.
        # By adding some padding and using the more central area, the image quality improves
        padding = 3
        for i in range(n_patches):
            p_left, p_right = 0, 0
            p_top, p_bottom = 0, 0
            x = b[i][0]
            y = b[i][1]
            if (x != 0):
                p_left = padding
            if (x + patch_size_lr != low_width):
                p_right = padding
            if (y != 0):
                p_top = padding
            if (y + patch_size_lr != low_height):
                p_bottom = padding
            left = scale_factor*x + p_left
            right = scale_factor * (x + patch_size_lr) - p_right
            top = scale_factor*y + p_top
            bottom = scale_factor * (y+patch_size_lr) - p_bottom

            left_lr = p_left
            right_lr = scale_factor*patch_size_lr - p_right
            top_lr = p_top
            bottom_lr = scale_factor*patch_size_lr - p_bottom

            hr_patch = hr_patches[i, :, left_lr:right_lr, top_lr:bottom_lr]
            hr_image[:, left:right, top:bottom] = hr_patch
    # im, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    # ax[0].imshow(hr_image.permute(1, 2, 0).numpy())
    # ax[1].imshow(original_image.permute(1, 2, 0).numpy())
    # plt.show()
    return hr_image


home = str(pathlib.Path.home())
image_path = str(pathlib.Path.home() /
                 ".keras/datasets/set5/test/set5/*.png")

datasets = ["set5", "set14", "Urban100", "Manga109", "BSDS100"]

dataset_paths = [
    f"{home}/.keras/datasets/set5/test/{d}/*.png" for d in datasets]

files = list(glob(dataset_paths[1]))

# Load the model
gen = HAN(num_resgroups=config.N_GROUPS, num_rcab=config.N_RCAB, height=config.LOW_RES,
          width=config.LOW_RES, in_channels=3, layer_channels=config.N_CHANNELS, scale_factor=config.SCALE_FACTOR).cuda()
opt_gen = optim.Adam(
    gen.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))

load_checkpoint(config.CHECKPOINT, gen, opt_gen, config.LEARNING_RATE)
gen.eval()

s_index = []
p_nr = []

# for f in files:
#     image = Image.open(f)
#     image = np.array(image)
#     if len(image.shape) == 2:
#         bw = True
#         image = np.stack([image, image, image], axis=2)
#     image = augmentations.highres_transform(image=image)["image"]
#     high_res = recreate_image(image, gen)
#     image = image.permute(1, 2, 0).numpy()
#     high_res = high_res.permute(1, 2, 0).numpy()
#     high_res = np.clip(high_res, 0, 1)
#     # print(high_res.min(), high_res.max())
#     # print(image.min(), image.max())
#     s_index.append(ssim(image, high_res, multichannel=True))
#     p_nr.append(psnr(image, high_res))
#     if np.any(np.isnan(s_index[-1])):
#         print(image.shape)

for x in range(len(dataset_paths)):
    files = list(glob(dataset_paths[x]))
    s_index = []
    p_nr = []

    # Computes SSIM and PSNR in the Y channel
    for f in files:
        image = Image.open(f)
        image = np.array(image)
        if len(image.shape) == 2:
            bw = True
            image = np.stack([image, image, image], axis=2)
        image = augmentations.highres_transform(image=image)["image"]
        high_res = recreate_image(image, gen, scale_factor=config.SCALE_FACTOR)
        image = image.permute(1, 2, 0).numpy()
        high_res = high_res.permute(1, 2, 0).numpy()
        high_res = np.clip(high_res, 0, 1)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)
        high_res = cv2.cvtColor(high_res, cv2.COLOR_RGB2YCR_CB)
        s_index.append(
            ssim(image[:, :, 0], high_res[:, :, 0], multichannel=False))
        p_nr.append(psnr(image, high_res))
        if np.any(np.isnan(s_index[-1])):
            print(image.shape)
    print(f"Dataset {datasets[x]}:")
    print(f"Average SSIM: {np.mean(s_index)}")
    print(f"Average PSNR: {np.mean(p_nr)}")

# hr = augmentations.highres_transform(image=image)["image"]
# hr = torch.reshape(hr, (1, 3, 96, 96)).cuda()
# low_res_input = augmentations.lowres_transform(
#     image=image)["image"].unsqueeze(0).to(config.DEVICE)

# # Load the generator model:

# generated = gen(low_res_input).cpu().detach().numpy()[0]
# # generated = config.test_transform(image=generated)
# # generated = np.einsum('abc->bca', generated)
# generated = generated.transpose(1, 2, 0)
# generated = np.clip(generated, 0, 1)
# # generated = generated*255
# # generated[generated <= 0] = 0
# # generated[generated >= 255] = 255
# # generated = generated.astype('uint8')

# lri = low_res_input.cpu().detach().numpy()[0]
# lri = np.einsum('abc->bca', lri)

# low_res = A.Resize(height=config.HIGH_RES, width=config.HIGH_RES,
#                    interpolation=Image.BICUBIC)(image=low_res)["image"]
