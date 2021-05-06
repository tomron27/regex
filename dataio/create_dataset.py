import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import BASE_DIR

train_root = [
    os.path.join(BASE_DIR, "datasets/BraTS18/train/LGG/"),
    os.path.join(BASE_DIR, "datasets/BraTS18/train/HGG/"),
]

dest_train = os.path.join(BASE_DIR, "datasets/BraTS18/train_proc_new/")


def dilute_and_save_image_and_mask(x, mask, dest, folder):
    """
    Skips slices without tumor and saves to .npz file
    """
    mask_filtered = []
    x_filtered = []
    start_idx = None
    for i in range(t1_img.shape[0]):
        area = mask[i].sum()
        if area > 0:
            if start_idx is None:
                start_idx = i
            mask_filtered.append(mask[i])
            x_filtered.append(x[:, i])
    x_filtered = np.array(x_filtered, dtype=np.float64)
    x_filtered = np.transpose(x_filtered, (1, 0, 2, 3))
    mask_filtered = np.array(mask_filtered, dtype=np.uint8)

    # Save slices as npy 2d arrays
    os.makedirs(os.path.join(dest, folder), exist_ok=True)
    for i in range(x_filtered.shape[1]):
        area = mask_filtered[i].sum()
        img_fname = os.path.join(dest, folder, folder + f"_slice={str(start_idx + i)}_y={area}.npz")
        mask_fname = os.path.join(dest, folder, folder + f"_slice={str(start_idx + i)}_mask.npz")
        # img_fname = os.path.join(dest, folder + f"_slice={str(start_idx + i)}_y={area}.npy")
        # mask_fname = os.path.join(dest, folder + f"_slice={str(start_idx + i)}_mask.npy")
        try:
            np.savez_compressed(img_fname, data=x_filtered[:, i])
            np.savez_compressed(mask_fname, data=mask_filtered[i])
            # np.save(img_fname, x_filtered[:, i])
            # np.save(mask_fname, mask_filtered[i])
        except:
            print(f"Error occurred on {img_fname}, continuing")


if __name__ == "__main__":
    for train_folder in train_root:
        print(f"Probing folder '{train_folder}'")
        folders = os.listdir(train_folder)
        os.makedirs(dest_train, exist_ok=True)
        for i, folder in tqdm(enumerate(folders), total=len(folders)):
            if "HGG" in train_folder and i in (50, 131):
                continue    # EOF bugs
            t1_img = nib.load(os.path.join(train_folder, folder, folder + "_t1.nii.gz")).get_fdata()
            t1_img = np.rot90(t1_img, k=1, axes=(0, 2))
            t1ce_img = nib.load(os.path.join(train_folder, folder, folder + "_t1ce.nii.gz")).get_fdata()
            t1ce_img = np.rot90(t1ce_img, k=1, axes=(0, 2))
            t2_img = nib.load(os.path.join(train_folder, folder, folder + "_t2.nii.gz")).get_fdata()
            t2_img = np.rot90(t2_img, k=1, axes=(0, 2))
            flair_img = nib.load(os.path.join(train_folder, folder, folder + "_flair.nii.gz")).get_fdata()
            flair_img = np.rot90(flair_img, k=1, axes=(0, 2))

            mask = nib.load(os.path.join(train_folder, folder, folder + "_seg.nii.gz")).get_fdata()
            mask = np.rot90(mask, k=1, axes=(0, 2))

            mask = (mask > 0).astype(np.uint8)
            x = np.zeros(shape=(4, *t1_img.shape), dtype=np.float64)
            x[0] = t1_img
            x[1] = t1ce_img
            x[2] = t2_img
            x[3] = flair_img

            dilute_and_save_image_and_mask(x, mask, dest_train, folder)


    # t2_img = nib.load(os.path.join(sample_dir, "Brats18_TCIA08_469_1_t2.nii.gz")).get_fdata()
    # t2_img = np.rot90(t2_img, k=1, axes=(0, 2))
    # mask = nib.load(os.path.join(sample_dir, "Brats18_TCIA08_469_1_seg.nii.gz")).get_fdata()
    # mask = np.rot90(mask, k=1, axes=(0, 2))
    # idx = 10
    # t2_img_slice = t2_img[idx]
    # mask_slice = mask[idx]
    # mask_slice = (mask_slice > 0).astype(np.uint8)
    #
    # plt.imshow(t2_img_slice, cmap="gray")
    # plt.imshow(mask_slice, cmap="jet", alpha=0.5)
    # plt.show()
    # print(mask_slice.sum())
    # pass