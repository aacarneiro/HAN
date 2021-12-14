from configparser import Interpolation
import torch
from PIL import Image
import pathlib
import numpy as np

LOAD_MODEL = True
SAVE_MODEL = True
# CHECKPOINT = "checkpoint/train.zip"
DEVICE = "cuda"
LEARNING_RATE = 1e-5
NUM_EPOCHS = 100
# If saving is activated, saves the model every SAVE_EVERY epochs
SAVE_EVERY = 3
BATCH_SIZE = 16
NUM_WORKERS = 0
HIGH_RES = 96
SCALE_FACTOR = 3
LOW_RES = HIGH_RES//SCALE_FACTOR
IMG_CHANNELS = 3

N_RCAB = 8
N_GROUPS = 8
N_CHANNELS = 64
KERNEL_SIZE = 9
CHECKPOINT = f"checkpoint/train_N={N_GROUPS}_m={N_RCAB}_C={N_CHANNELS}_k={KERNEL_SIZE}_x{SCALE_FACTOR}.zip"

# Use the script compute_mean_std.py to calculate the mean and std of the training dataset
MEAN = np.array([0, 0, 0])
STD = np.array([1, 1, 1])

# MEAN = np.array([0.45069343, 0.43524462, 0.40086321])
# STD = np.array([0.28370893, 0.2684125, 0.2890807])

DATA_PATH = str(pathlib.Path.home() /
                ".keras/datasets/DIV2k/DIV2K_proc2/*.png")
