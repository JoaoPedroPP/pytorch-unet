import os

from torch.utils.data import Dataset
from torchvision.io import read_image

class LIDCDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.input = []
        self.mask = []

        imgs = lis

    def __getitem__(self, idx):
        image = self.input[idx]
        mask = self.mask[idx]

        return [image, mask]
