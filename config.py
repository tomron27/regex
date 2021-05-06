import os
import json
from torchvision.transforms import Compose
# from kornia.geometry import Resize
# from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomAffine, RandomErasing, RandomPerspective
# from src.dataio.transforms import Normalize, CustomBrightness, CustomContrast, RandomContrast
BASE_DIR = "/mnt/ml-srv1/home/tomron27/"

class TrainConfig():
    def __init__(self):
        self.config = {
            "data_path": os.path.join(BASE_DIR, "datasets/BraTS18/train_proc/"),
            "log_path": os.path.join(BASE_DIR, "projects/regex/logs/"),
            "gpu_id": 3,
            "weights": None,
            "freeze_backbone": False,
            "seed": 42,
            "num_epochs": 100,
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
            "learnable_attn": True,
            "learnable_lamb": False,
            "attn_kl": True,
            "kl_weight": 10.0,
            "init_lamb": 0.0,
            # "loss_weights" : [0.1, 0.2, 0.7],
            "loss_weights": True,
            "name": "resnet50_regression_baseline",
            "group": "baseline",
            "arch": "resnet50_regressor",
            "channels": 4,
            "width": 1,
            "save_metric": "rmse",
            # l2x params
            "tau": 0.5,
            "k": 256,
            "spatial_dim": 512,
            "factor": 16,
            # Augmentations
            # "transforms": Compose([
            #     RandomResizedCrop(p=1.0, size=(512, 512), scale=(0.8, 3.3)),
            #     RandomHorizontalFlip(p=0.5),
            #     RandomVerticalFlip(p=0.5),
            #     CustomContrast(p=1.0, contrast=(1.0, 1.1)),
            #     CustomBrightness(p=1.0, brightness=(0.3, 1.1)),
            # ]),
            "bad_files": [
                            "Brats18_CBICA_AWH_1_slice=69_y=2501.npz",
                            "Brats18_CBICA_AWH_1_slice=69_mask.npz",
                            "Brats18_CBICA_AWH_1_slice=73_y=1726.npz",
                            "Brats18_CBICA_AWH_1_slice=73_mask.npz",
                            "Brats18_CBICA_ATV_1_slice=72_y=1093.npz",
                            "Brats18_CBICA_ATV_1_slice=72_mask.npz"
                            ]
        }