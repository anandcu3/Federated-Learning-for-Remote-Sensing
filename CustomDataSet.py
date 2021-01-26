from torch.utils.data import Dataset
from natsort import natsorted
from PIL import Image
import numpy as np
import torch
import glob
import os


class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform, labelmat):
        self.main_dir = main_dir
        self.transforms = transform
        self.all_imgs = glob.glob(os.path.join(
            main_dir, '**/*.tif'), recursive=False)
        self.total_imgs = natsorted(self.all_imgs)
        self.xlabels = labelmat

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        # print(idx,len(self.total_imgs))
        img_loc = self.total_imgs[idx]
        # print(img_loc)
        imagebaselabel = os.path.splitext(os.path.basename(img_loc))[0]
        label = self.xlabels[np.where(
            self.xlabels[:, 0] == imagebaselabel), 1:].reshape(17).astype(np.int64)
        # print(label,label.shape)
        tensor_label = torch.from_numpy(label)
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transforms(image)
        return tensor_image, tensor_label

    def __getlabel__(self, idx):
        img_loc = self.total_imgs[idx]
        # print(img_loc)
        imagebaselabel = os.path.splitext(os.path.basename(img_loc))[0]
        label = self.xlabels[np.where(
            self.xlabels[:, 0] == imagebaselabel), 1:].reshape(17).astype(np.int64)

        return label
