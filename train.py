from torchsummary import summary
import torch
import config
from torch import nn
from torch import optim
from utils import load_checkpoint, optimizer_to, save_checkpoint, freeze_rcan
from torch.utils.data import DataLoader
from model import HAN
from dataset import MyImageFolder
import os
import pathlib
from tqdm import tqdm
from time import time

torch.backends.cudnn.benchmark = True


def train(loader, gen, opt, loss, accum_grad=1, grad_clipping=True):
    """[summary]

    Args:
        loader ([type]): [description]
        gen ([type]): [description]
        opt ([type]): [description]
        loss ([type]): [description]
        accum_grad (int, optional): [description]. Number of steps to accumulate the gradients. If set to 1, gradient step is run for every batch.
        grad_clipping (bool, optional): [description]. Whether to clip the gradients.
    """
    # loop = tqdm(loader, leave=True)
    total_time = 0
    time_load = 0
    time_grad = 0
    total_loss = 0

    # Only runs the optimizer step after accum_iter iterations
    accum_iter = accum_grad
    for idx, (low_res, high_res) in enumerate(loader):
        # t0 = time()
        high_res = high_res.to(config.DEVICE)
        low_res = low_res.to(config.DEVICE)
        # t1 = time()
        generated = gen(low_res)
        image_loss = loss(generated, high_res)
        # t2 = time()
        image_loss.backward()
        # t3 = time()
        if ((idx + 1) % accum_iter == 0) or (idx + 1 == len(loader)):
            nn.utils.clip_grad.clip_grad_norm_(gen.parameters(), 10)
            opt.step()
            opt.zero_grad()
        # t4 = time()
        total_loss += image_loss
        # print(total_loss.detach().cpu().numpy(), generated.detach().cpu().numpy(
        # ).min(), generated.detach().cpu().numpy().max())

        # time_grad += t4 - t3
        # total_time += t4 - t0
        # time_load += t1 - t0

        # if idx % 200 == 0:
        #     plot_examples("data/BSDS300/images", gen)
    total_loss = total_loss*256/(len(loader)*3*config.HIGH_RES**2)
    print(
        f"Total loss: {total_loss:.5}.\n")
    # print(
    #     f"Total time: {total_time:.2}s. Load: {time_load:.2}. Grad: {time_grad:.2}")


dataset_path = config.DATA_PATH

dataset = MyImageFolder(path=dataset_path)
# dataset = MyImageFolder(
#     root_dir="E:\\Code\\machine_learning\\SRGAN\\data\\DIV2K")
loader = DataLoader(dataset,
                    batch_size=config.BATCH_SIZE,
                    shuffle=True,
                    pin_memory=True)
gen = HAN(num_resgroups=config.N_GROUPS, num_rcab=config.N_RCAB, height=config.LOW_RES,
          width=config.LOW_RES, in_channels=3, layer_channels=config.N_CHANNELS, scale_factor=config.SCALE_FACTOR)
gen = gen.to(config.DEVICE)
# freeze_rcan(gen)
# opt = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE,
#                  betas=(0.9, 0.999))
opt = optim.SGD(gen.parameters(), lr=config.LEARNING_RATE, momentum=0.1)
# opt = optim.RMSprop(gen.parameters(), lr=config.LEARNING_RATE, momentum=0.7)
# optimizer_to(opt, config.DEVICE)
mse = nn.MSELoss(reduction="sum").to(config.DEVICE)
l1loss = nn.L1Loss(reduction="sum").to(config.DEVICE)

# summary(gen, (3, 48, 48), device="cuda")

if config.LOAD_MODEL:
    load_checkpoint(config.CHECKPOINT,
                    gen,
                    opt,
                    config.LEARNING_RATE)

for epoch in range(1, config.NUM_EPOCHS+1):
    train(loader, gen, opt, mse, accum_grad=2, grad_clipping=True)

    if (epoch % config.SAVE_EVERY == 0 or epoch == config.NUM_EPOCHS) and config.SAVE_MODEL:
        save_checkpoint(gen, opt, epoch, filename=config.CHECKPOINT)
