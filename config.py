import json
from torchvision.transforms import Compose
# from kornia.geometry import Resize
# from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomAffine, RandomErasing, RandomPerspective
# from src.dataio.transforms import Normalize, CustomBrightness, CustomContrast, RandomContrast


class TrainConfig():
    def __init__(self):
        self.config = {
            "data_path": "/data/home/tomron27/datasets/BraTS18/train_proc/",
            "log_path": "/data/home/tomron27/projects/regex/logs/",
            "gpu_id": 1,
            "weights": None,
            "freeze_backbone": True,
            "seed": 42,
            "num_epochs": 100,
            "chekpoint_save_interval": 100,
            "min_epoch_save": 10,
            "batch_size": 32,
            "num_workers": 4,
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
        }