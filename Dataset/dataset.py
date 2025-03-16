import os

from torch.utils.data import Dataset
from torchvision.io import decode_image

class LIDCDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        raw = list(filter(lambda l: not l.endswith('edge_mask.png'), os.listdir(img_dir)))
        self.input = list(filter(lambda l: not l.endswith('mask.png'), raw))
        self.mask = list(filter(lambda l: l.endswith('mask.png'), raw))

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        image = decode_image(os.path.join(self.img_dir, self.input[idx]), mode='GRAY')
        if self.transform:
            image = self.transform(image)
        mask = decode_image(os.path.join(self.img_dir, self.mask[idx]), mode='GRAY').div(255)
        if self.target_transform:
            mask = self.target_transform(mask)

        return [image, mask]
