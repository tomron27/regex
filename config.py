import os
import torch
from torchvision.transforms import Compose
from kornia.augmentation import RandomAffine, Normalize, RandomVerticalFlip, RandomHorizontalFlip
from kornia.geometry.transform import Resize
from dataio.augmentations import CustomBrightness, CustomContrast

BASE_DIR = "/hdd0"
GPU_ID = "0"

class TrainConfigViT():
    def __init__(self):
        self.config = {
            "data_path": os.path.join(BASE_DIR, "datasets/BraTS18/train_split_proc_new"),
            "log_path": os.path.join(BASE_DIR, "projects/regex/logs/"),
            "gpu_id": GPU_ID,
            "weights": None,
            "pretrained": True,
            # "weights": os.path.join(BASE_DIR, "projects/regex/logs/vit_baseline/20210520_17:29:31/vit_baseline__best__epoch=026_score=0.9759.pt"),
            "freeze_backbone": False,
            "prefetch_data": False,
            "num_classes": 2,
            "subsample_frac": 1.0,
            "seed": 42,
            "num_epochs": 20,
            "chekpoint_save_interval": 10,
            "min_epoch_save": 1,
            "batch_size": 16,
            "num_workers": 8,
            "lr": 1e-4,
            "optim_step": 100,
            "optim_factor": 0.5,
            "train_frac": 0.8,
            "use_gpu": True,
            "balanced_sampler": False,
            "patch_size": 16,
            "dim": 768,
            "depth": 12,
            "heads": 12,
            "mlp_dim": 1024,
            "dropout": 0.0,
            "emb_dropout": 0.0,
            "name": "vit_baseline",
            "group": "vit",
            "channels": 4,
            "width": 1,
            "save_metric": "balanced_accuracy_score",
            "spatial_dim": 256,
            # Augmentations
            "train_transforms": Compose([
                Normalize(mean=torch.tensor([149.56119, 165.83559, 166.13501, 112.61901]),
                          std=torch.tensor([636.8766, 653.8386, 759.8256, 718.83594])),
                Resize((256, 256)),
                RandomVerticalFlip(p=0.5),
                RandomHorizontalFlip(p=0.5),
                RandomAffine(p=1.0, degrees=(-180, 180),
                             translate=(0.2, 0.2),
                             scale=(0.9, 1.5),
                             shear=(0.7, 1.5)),
            ]),
            "test_transforms": Compose([
                Normalize(mean=torch.tensor([149.56119, 165.83559, 166.13501, 112.61901]),
                          std=torch.tensor([636.8766, 653.8386, 759.8256, 718.83594])),
                Resize((256, 256)),
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


class TrainConfig():
    def __init__(self):
        self.config = {
            "data_path": os.path.join(BASE_DIR, "datasets/BraTS18/train_split_proc_new"),
            "log_path": os.path.join(BASE_DIR, "projects/regex/logs/"),
            "gpu_id": GPU_ID,
            # "weights": None,
            "weights": "/hdd0/projects/regex/models/unet_encoder_baseline_score=0.9717.pt",
            "freeze_backbone": True,
            "prefetch_data": False,
            "num_classes": 2,
            "subsample_frac": 1.0,
            "seed": 42,
            "num_epochs": 20,
            "chekpoint_save_interval": 10,
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
            "learnable_marginals": False,
            "learnable_lamb": False,
            "attn_kl": False,
            "kl_weight": 10.0,
            "detach_targets": False,
            "init_lamb": 0.0,
            # "loss_weights" : [0.1, 0.2, 0.7],
            # "loss_weights": True,
            "name": "unet_encoder_tau4_ensemble",
            "group": "tau4_ensemble",
            "arch": "unet_encoder_classifier",
            "channels": 4,
            "init_features": 32,
            "attn_embed_factor": 16,
            "width": 1,
            "save_metric": "balanced_accuracy_score",
            "spatial_dim": 256,
            # Augmentations
            "train_transforms": Compose([
                Normalize(mean=torch.tensor([149.56119, 165.83559, 166.13501, 112.61901]),
                          std=torch.tensor([636.8766, 653.8386, 759.8256, 718.83594])),
                Resize((256, 256)),
                RandomVerticalFlip(p=0.5),
                RandomHorizontalFlip(p=0.5),
                RandomAffine(p=1.0, degrees=(-180, 180),
                             translate=(0.2, 0.2),
                             scale=(0.9, 1.5),
                             shear=(0.7, 1.5)),
            ]),
            "test_transforms": Compose([
                Normalize(mean=torch.tensor([149.56119, 165.83559, 166.13501, 112.61901]),
                          std=torch.tensor([636.8766, 653.8386, 759.8256, 718.83594])),
                Resize((256, 256)),
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