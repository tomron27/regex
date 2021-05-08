import os
import torch
from torchvision.transforms import Compose
from kornia.augmentation import RandomAffine, Normalize, RandomVerticalFlip, RandomHorizontalFlip
from dataio.augmentations import CustomBrightness, CustomContrast

BASE_DIR = "/mnt/ml-srv1/home/tomron27/"


class TrainConfig():
    def __init__(self):
        self.config = {
            "data_path": os.path.join(BASE_DIR, "datasets/BraTS18/train_split_proc"),
            "log_path": os.path.join(BASE_DIR, "projects/regex/logs"),
            "gpu_id": 3,
            "weights": None,
            "freeze_backbone": False,
            "prefetch_data": False,
            "num_classes": 2,
            "subsample_frac": 1.0,
            "seed": 42,
            "num_epochs": 500,
            "chekpoint_save_interval": 100,
            "min_epoch_save": 1,
            "batch_size": 64,
            "num_workers": 8,
            "lr": 1e-3,
            "optim_step": 100,
            "optim_factor": 0.5,
            "train_frac": 0.8,
            "use_gpu": True,
            "balanced_sampler": False,
            "learnable_attn": False,
            "learnable_lamb": False,
            "attn_kl": True,
            "kl_weight": 10.0,
            "init_lamb": 0.0,
            # "loss_weights" : [0.1, 0.2, 0.7],
            # "loss_weights": True,
            "name": "resnet50_binary_classification",
            "group": "baseline",
            "arch": "resnet50_classifier",
            "channels": 4,
            "init_features": 32,
            "width": 1,
            "save_metric": "balanced_accuracy_score",
            # l2x params
            "tau": 0.5,
            "k": 64,
            "spatial_dim": 240,
            "factor": 16,
            # Augmentations
            "train_transforms": Compose([
                Normalize(mean=torch.tensor([149.56119, 165.83559, 166.13501, 112.61901]),
                          std=torch.tensor([636.8766, 653.8386, 759.8256, 718.83594])),
                RandomVerticalFlip(p=0.5),
                RandomHorizontalFlip(p=0.5),
            ]),
            "test_transforms": Compose([
                Normalize(mean=torch.tensor([149.56119, 165.83559, 166.13501, 112.61901]),
                          std=torch.tensor([636.8766, 653.8386, 759.8256, 718.83594])),
            ]),
            "bad_files": [
                            "Brats18_CBICA_AWH_1_slice=69_y=2501.npz",
                            "Brats18_CBICA_AWH_1_slice=69_mask.npz",
                            "Brats18_CBICA_AWH_1_slice=73_y=1726.npz",
                            "Brats18_CBICA_AWH_1_slice=73_mask.npz",
                            "Brats18_CBICA_ATV_1_slice=72_y=1093.npz",
                            "Brats18_CBICA_ATV_1_slice=72_mask.npz"
                            ]
        }