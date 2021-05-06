import os
import re
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from config import BASE_DIR


def probe_data_folder(folder, train_frac=0.8, random_state=42, bad_files=None):
    all_data = os.listdir(folder)
    if bad_files is not None:
        all_data = [x for x in all_data if x not in bad_files]
    images_fnames = sorted([x for x in all_data if "mask" not in x])
    masks_fnames = sorted([x for x in all_data if "mask" in x])
    all_data_pairs = list(zip(images_fnames, masks_fnames))
    random.Random(random_state).shuffle(all_data_pairs)
    split_idx = int(train_frac * len(all_data_pairs))
    train, test = all_data_pairs[:split_idx], all_data_pairs[split_idx:]
    return train, test


class BraTS18(Dataset):
    def __init__(self, base_folder, data_list, transforms=None, get_mask=False, prefetch_data=False):
        super(BraTS18, self).__init__()
        self.base_folder = base_folder
        self.data_list = data_list
        self.transforms = transforms
        self.get_mask = get_mask
        self.prefetch_data = prefetch_data
        if self.prefetch_data:
            print("Prefetching dataset")
            self.data = []
            for i, (image_fname, mask_fname) in tqdm(enumerate(self.data_list), total=len(self.data_list)):
                image, label = self.get_image_and_mask(image_fname)
                self.data.append((image, label))

    def get_image_and_mask(self, image_fname, mask_fname=None):
        try:
            image = np.load(os.path.join(self.base_folder, image_fname))['data']
            image = torch.tensor(image, dtype=torch.float32)
        except:
            print(f"Error encountered on '{image_fname}'; '{mask_fname}'")
            raise ValueError
        label = int(re.search(r"y=([0-9]+)", image_fname).groups(1)[0])
        label = torch.tensor(label, dtype=torch.float32)

        if self.get_mask:
            mask = np.load(os.path.join(self.base_folder, mask_fname))['data']
            mask = torch.tensor(mask, dtype=torch.float32)
            return image, (label, mask)

        return image, label

    def __getitem__(self, index):
        if self.prefetch_data:
            image, label = self.data[index]
        else:
            image_fname, mask_fname = self.data_list[index]
            mask_fname = mask_fname if self.get_mask else None
            image, label = self.get_image_and_mask(image_fname, mask_fname=mask_fname)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, label

    def __len__(self):
        return len(self.data_list)

# if __name__ == "__main__":
#     folder = os.path.join(BASE_DIR, "datasets/BraTS18/train_proc/")
#     train_metadata, test_metadata = probe_data_folder(folder)
#     train_data = BraTS18(folder, train_metadata)
#     image, mask = train_data.__getitem__(0)
