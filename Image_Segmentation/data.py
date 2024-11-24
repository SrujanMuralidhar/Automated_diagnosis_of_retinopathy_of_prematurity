import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class DriveDataset(Dataset):
    def __init__(self, images_path, masks_path, size=(512, 512)):
        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(images_path)
        self.size = size

    def __getitem__(self, index):
        """ Reading and resizing image """
        image = Image.open(self.images_path[index]).convert('RGB')
        image = image.resize(self.size)
        image = np.array(image) / 255.0  # Normalize to [0, 1]
        image = np.transpose(image, (2, 0, 1))  # (3, 512, 512)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        """ Reading and resizing mask """
        mask = Image.open(self.masks_path[index]).convert('L')
        mask = mask.resize(self.size)
        mask = np.array(mask) / 255.0  # Normalize to [0, 1]
        mask = np.expand_dims(mask, axis=0)  # (1, 512, 512)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)

        return image, mask

    def __len__(self):
        return self.n_samples
