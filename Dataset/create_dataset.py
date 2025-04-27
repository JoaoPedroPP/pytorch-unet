import numpy as np
import os
from os import listdir

from PIL import Image

def main(data_dir, output_dir):
    gt_raw = list(filter(lambda x: x.endswith('png') and not x.endswith('mask.png'),listdir(data_dir)))
    # masks_raw = list(filter(lambda x: x.endswith('mask.png') and not x.endswith('edge_mask.png'),listdir(data_dir)))
    # edge_mask_raw = list(filter(lambda x: x.endswith('edge_mask.png'),listdir(data_dir)))

    for i, name in enumerate(gt_raw):
        filename, _extension = name.split('.')

        # using PIL
        gt_img = Image.open(os.path.join(data_dir, name)).convert('L')
        mask_img = Image.open(f"{os.path.join(data_dir, filename)}_mask.png").convert('L')
        edge_mask_img = Image.open(f"{os.path.join(data_dir,filename)}_edge_mask.png").convert('L')
        gt = np.array(gt_img)
        mask = np.array(mask_img)
        edge = np.array(edge_mask_img)
        merge = np.array([gt, mask, edge])
        # Aqui o array sai (3x128x96)
        # print(merge)
        print(merge.shape)

        # Aqui os array sai 128x96x3
        # merge = np.ravel(merge, order='F')
        # merge = np.reshape(merge, (128, 96, 3))
        # print(merge)
        # print(merge.shape)

        np.save(f'{os.path.join(output_dir, filename)}.npy', merge, False)
        print(f"{filename}: Done")


main('../support_images/dataset/raw/', '../support_images/dataset/raw2')
# main("/run/media/jpolonip/JP2-HD/MestradoFiles/Dataset/raw/train", "/run/media/jpolonip/JP2-HD/MestradoFiles/Dataset/raw_gt+edge+mask/train")
