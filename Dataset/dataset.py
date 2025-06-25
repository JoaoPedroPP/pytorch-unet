import numpy as np
from torch.utils.data import Dataset
from torchvision.io import decode_image

class LIDCDataset(Dataset):
    def __init__(self, imgs, masks, transform=None, target_transform=None):
        self.imgs = imgs
        self.masks = masks
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # For regular input images
        # image = decode_image(self.imgs[idx], mode='GRAY')

        # For npy 2 dim images
        image = np.load(self.imgs[idx])
        if self.transform:
            image = self.transform(image)
        mask = decode_image(self.masks[idx], mode='GRAY').div(255)
        if self.target_transform:
            mask = self.target_transform(mask)

        return [image, mask]
