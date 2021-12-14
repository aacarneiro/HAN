from PIL import Image
import numpy as np
from glob import glob
import config


# Computes the mean and std of the dataset you're training on.
# Add the values to the config.py file.
files = glob(config.DATA_PATH)
ims = []

for f in files:
    img = Image.open(f)
    img = np.array(img)
    ims.append(img)
ims = np.array(ims) / 255
# Computes the mean of all images,
# then the mean across the height,
# then the mean across the width.
mean = ims.mean(axis=0).mean(axis=0).mean(axis=0)
std = ims.std(axis=0).mean(axis=0).mean(axis=0)
