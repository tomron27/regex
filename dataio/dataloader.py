import os
import re
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import Dataset


def probe_data_folder(folder, train_frac=0.8, random_state=42):
    all_data = os.listdir(folder)
    images_fnames = sorted([x for x in all_data if "mask" not in x])
    masks_fnames = sorted([x for x in all_data if "mask" in x])
    all_data_pairs = list(zip(images_fnames, masks_fnames))
    random.Random(random_state).shuffle(all_data_pairs)
    split_idx = int(train_frac * len(all_data_pairs))
    train, test = all_data_pairs[:split_idx], all_data_pairs[split_idx:]
    return train, test


class BraTS18(Dataset):
    def __init__(self, base_folder, data_list, transforms=None, get_mask=False):
        super(BraTS18, self).__init__()
        self.base_folder = base_folder
        self.data_list = data_list
        self.transforms = transforms
        self.get_mask = get_mask

    def __getitem__(self, index):
        image_fname, mask_fname = self.data_list[index]
        image = np.load(os.path.join(self.base_folder, image_fname))['data']
        image = torch.tensor(image, dtype=torch.float32)
        if self.get_mask:
            mask = np.load(os.path.join(self.base_folder, mask_fname))['data']
        label = int(re.search(r"y=([0-9]+)", image_fname).groups(1)[0])
        label = torch.tensor(label, dtype=torch.float32)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, label

    def __len__(self):
        return len(self.data_list)

if __name__ == "__main__":
    folder = "/data/home/tomron27/datasets/BraTS18/train_proc/"
    train_metadata, test_metadata = probe_data_folder(folder)
    train_data = BraTS18(folder, train_metadata)
    image, mask = train_data.__getitem__(0)
