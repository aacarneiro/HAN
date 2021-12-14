import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import config


highres_transform = A.Compose([A.Normalize(mean=config.MEAN,
                                           std=config.STD),
                               ToTensorV2()])

lowres_transform = A.Compose([A.Resize(width=config.LOW_RES, height=config.LOW_RES, interpolation=Image.BICUBIC),
                              A.Normalize(mean=config.MEAN,
                                          std=config.STD),
                              ToTensorV2()])

lowres_transform_notensor = A.Compose([A.Resize(width=config.LOW_RES, height=config.LOW_RES, interpolation=Image.BICUBIC),
                                       A.Normalize(mean=config.MEAN,
                                                   std=config.STD)])

both_transforms = A.Compose([A.RandomCrop(width=config.HIGH_RES, height=config.HIGH_RES),
                             A.HorizontalFlip(p=0.5),
                             A.RandomRotate90(p=0.5)])

test_transform = A.Compose([A.Normalize(mean=config.MEAN,
                                        std=config.STD),
                            ToTensorV2()])
