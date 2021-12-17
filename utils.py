import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from glob import glob
import cv2
from PIL import Image

import config
import augmentations


def save_checkpoint(model, optimizer, epoch, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    # checkpoint["state_dict"].pop("final.bias", None)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    # optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging
    # for param_group in optimizer.param_groups:
    #     param_group["lr"] = lr


def upscale_image(image, gen, scale_factor):
    """Do not rescale the input
    """
    original_image = image
    patch_size_lr = config.LOW_RES
    channels, low_width, low_height = image.shape
    width = scale_factor*low_width
    height = scale_factor*low_height
    hr_image = torch.empty(channels, width, height)
    # image = transforms.Resize((low_width, low_height))(image)
    # image = augmentations.highres_transform(image=image)["image"]
    # image = torch.from_numpy(image).permute(2, 0, 1)
    nx_patches = int(low_width*1.25/patch_size_lr)+1
    ny_patches = int(low_height*1.25/patch_size_lr)+1
    x_steps = np.linspace(0, low_width, nx_patches,
                          dtype=np.int, endpoint=False)
    y_steps = np.linspace(0, low_height, ny_patches,
                          dtype=np.int, endpoint=False)
    x_steps[-1] = low_width-patch_size_lr
    y_steps[-1] = low_height-patch_size_lr

    # Uses batches to call the generator fewer times
    n_patches = nx_patches*ny_patches
    batch_size = 32
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
        padding = 4
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
    hr = hr_image.permute(1, 2, 0).numpy()
    lr = original_image.permute(1, 2, 0).numpy()
    lr = cv2.resize(lr, (width, height))

    # Plot the images
    im, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    ax[0].imshow(lr)
    ax[1].imshow(hr)
    plt.show()
    return hr_image


def upscale_folder(model, path="image/*.png"):
    paths = glob(path)
    for p in paths:
        image = Image.open(p)
        image = np.array(image)
        if len(image.shape) == 2:
            bw = True
            image = np.stack([image, image, image], axis=2)
        image = augmentations.highres_transform(image=image)["image"]
        high_res = upscale_image(image, model, config.SCALE_FACTOR)
        image = image.permute(1, 2, 0).numpy()
        high_res = high_res.permute(1, 2, 0).numpy()
        high_res = np.clip(high_res, 0, 1)*255
        # Converts to BGR before saving, otherwise openCV switches the blue and red channel
        high_res = cv2.cvtColor(high_res, cv2.COLOR_RGB2BGR)
        hr = cv2.imwrite(f"{p[:-3]}_sr.png", high_res)


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
