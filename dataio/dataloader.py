import os
import re
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from config import BASE_DIR

from torchvision.transforms import Compose
from kornia.augmentation import RandomAffine, RandomVerticalFlip, RandomHorizontalFlip
from dataio.augmentations import CustomBrightness, CustomContrast


def get_patient_files(base_dir, patient, bad_files=None):
    files = os.listdir(os.path.join(base_dir, patient))
    if bad_files is not None:
        files = [x for x in files if x not in bad_files]
    images_fnames = sorted([os.path.join(base_dir, patient, x) for x in files if "mask" not in x])
    masks_fnames = sorted([os.path.join(base_dir, patient, x) for x in files if "mask" in x])
    return list(zip(images_fnames, masks_fnames))


def probe_data_folder(folder, train_frac=0.8, random_state=42, bad_files=None, subsample_frac=1.0):
    patient_folders = os.listdir(folder)
    random.Random(random_state).shuffle(patient_folders)
    if subsample_frac < 1.0:
        # Smaller dataset for fast prototyping
        print(f"Subsampling dataset by {subsample_frac}")
        subsample_idx = int(subsample_frac * len(patient_folders))
        patient_folders = patient_folders[:subsample_idx]
    split_idx = int(train_frac * len(patient_folders))
    train_patients, test_patients = patient_folders[:split_idx], patient_folders[split_idx:]
    train, test = [], []
    for patient in train_patients:
        train += get_patient_files(folder, patient, bad_files=bad_files)
    for patient in test_patients:
        test += get_patient_files(folder, patient, bad_files=bad_files)
    print(f"  Total of {len(train_patients)} train patients, {len(test_patients)} test patients")
    print(f"  Total of {len(train)} train slices, {len(test)} test slices")
    return train, test


class BraTS18(Dataset):
    def __init__(self, base_folder, data_list, transforms=None, shuffle=True, random_state=42, get_mask=False, prefetch_data=False):
        super(BraTS18, self).__init__()
        self.base_folder = base_folder
        self.transforms = transforms
        self.get_mask = get_mask
        self.prefetch_data = prefetch_data
        self.shuffle = shuffle
        self.random_state = random_state
        self.data_list = data_list
        if self.shuffle:
            random.Random(self.random_state).shuffle(self.data_list)
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
            image = self.transforms(image).squeeze()

        return image, label

    def __len__(self):
        return len(self.data_list)


class BraTS18Binary(Dataset):
    def __init__(self, base_folder, data_list, transforms=None, shuffle=True, random_state=42, get_mask=False, prefetch_data=False):
        super(BraTS18Binary, self).__init__()
        self.base_folder = base_folder
        self.transforms = transforms
        self.get_mask = get_mask
        self.prefetch_data = prefetch_data
        self.shuffle = shuffle
        self.random_state = random_state
        self.data_list = data_list
        if self.shuffle:
            random.Random(self.random_state).shuffle(self.data_list)
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
        label = torch.tensor(label, dtype=torch.long)

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
            image = self.transforms(image).squeeze()

        return image, label

    def __len__(self):
        return len(self.data_list)


if __name__ == "__main__":
    folder = os.path.join(BASE_DIR, "datasets/BraTS18/train_split_proc/")
    train_metadata, test_metadata = probe_data_folder(folder)
    # Transforms
    transforms = Compose([
        RandomVerticalFlip(p=0.5),
        RandomHorizontalFlip(p=0.5),
        # RandomAffine(p=1.0, degrees=(-180, 180),
        #              translate=(0.1, 0.1),
        #              scale=(0.7, 1.3),
        #              shear=(0.9, 1.1)),
        # CustomContrast(p=1.0, contrast=(0.7, 1.3)),
        # CustomBrightness(p=1.0, brightness=(0.3, 1.1)),
    ])
    # sub_data = train_metadata[:10]
    train_data = BraTS18Binary(folder, train_metadata, transforms=transforms, shuffle=True)
    # all_data = BraTS18(folder, sub_data, transforms=transforms, shuffle=True)
    import matplotlib.pyplot as plt
    for i in range(10):
        image, mask = train_data.__getitem__(i)
        plt.imshow(image[2], cmap="gray"); plt.title(mask.item()), plt.show()
    pass
