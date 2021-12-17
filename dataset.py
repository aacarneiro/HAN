import os
import numpy as np
import augmentations
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from glob import glob


class MyImageFolder(Dataset):
    def __init__(self, path):
        super(MyImageFolder, self).__init__()
        self.data = []
        self.data = glob(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = Image.open(self.data[index])
        image = np.array(image)
        image = augmentations.both_transforms(image=image)["image"]
        high_res = augmentations.highres_transform(image=image)["image"]
        low_res = augmentations.lowres_transform(image=image)["image"]
        return low_res, high_res


def test():
    dataset = MyImageFolder(root_dir="new_data/")
    loader = DataLoader(dataset, batch_size=1, num_workers=8)

    for low_res, high_res in loader:
        print(low_res.shape)
        print(high_res.shape)


if __name__ == "__main__":
    test()
