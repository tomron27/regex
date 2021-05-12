import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from datetime import datetime
import pickle

from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torch.utils.data import DataLoader

import config
from config import TrainConfig, BASE_DIR, GPU_ID
from dataio.dataloader import probe_data_folder, BraTS18Binary
from train_utils import log_stats_classification, write_stats_classification
from loss import TauKLDivLoss, MarginalPenaltyLoss, MarginalsExtendedLoss
from models.resnet import get_resnet50_attn_classifier
from models.unet import get_unet_regressor, get_unet_encoder_classifier

# Ignore pytorch warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID


def train(seed=None):
    raw_params = TrainConfig()
    # Serialize transforms
    train_transforms = raw_params.config.pop("train_transforms")
    test_transforms = raw_params.config.pop("test_transforms")
    raw_params.config["train_transforms"] = {"transforms": train_transforms.__str__()}
    raw_params.config["test_transforms"] = {"test_transforms": test_transforms.__str__()}

    params = raw_params.config

    # Set seed
    if seed is not None:
        params["seed"] = seed
        random.seed(params["seed"])
        np.random.seed(params["seed"])
        torch.manual_seed(params["seed"])

    wandb.init(project='regex',
               entity='tomron27',
               job_type="eval",
               reinit=True,
               config=params,
               notes=params["name"],
               group=params["group"])

    train_metadata, val_metadata, class_counts = probe_data_folder(params["data_path"],
                                                                   train_frac=params["train_frac"],
                                                                   bad_files=params["bad_files"],
                                                                   subsample_frac=params["subsample_frac"],
                                                                   count_classes=True)
    ### DEBUG
    # train_metadata, val_metadata = train_metadata[:128], val_metadata[:128]

    # Datasets
    train_dataset = BraTS18Binary(params["data_path"],
                            train_metadata,
                            transforms=train_transforms,
                            shuffle=True,
                            random_state=params["seed"],
                            prefetch_data=params["prefetch_data"])
    val_dataset = BraTS18Binary(params["data_path"],
                          val_metadata,
                          transforms=test_transforms,
                          prefetch_data=params["prefetch_data"],
                          shuffle=False)

    # Dataloaders
    train_loader = DataLoader(dataset=train_dataset,
                              num_workers=params["num_workers"],
                              pin_memory=True,
                              batch_size=params["batch_size"])
    val_loader = DataLoader(dataset=val_dataset,
                            num_workers=params["num_workers"],
                            pin_memory=True,
                            batch_size=params["batch_size"],
                            shuffle=False)

    # Model
    # model = get_resnet50_attn_regressor(**params)
    # model = get_unet_regressor(**params)
    # model = get_resnet50_attn_classifier(**params)
    model = get_unet_encoder_classifier(**params)

    # Create log dir
    log_dir = os.path.join(params["log_path"], params["name"], datetime.now().strftime("%Y%m%d_%H:%M:%S"))
    os.makedirs(log_dir)
    print("Log dir: '{}' created".format(log_dir))
    pickle.dump(params, open(os.path.join(log_dir, "params.p"), "wb"))

    # CUDA
    device = torch.device("cuda" if torch.cuda.is_available() and params["use_gpu"] else "cpu")
    model = model.to(device)

    # Loss
    # criterion = torch.nn.MSELoss()
    # criterion = torch.nn.CrossEntropyLoss()
    # criterion = TauKLDivLoss(attn_kl=params["attn_kl"],
    #                          kl_weight=params["kl_weight"],
    #                          detach_targets=params["detach_targets"])
    # criterion = MarginalPenaltyLoss(attn_kl=params["attn_kl"], kl_weight=params["kl_weight"])
    criterion = MarginalsExtendedLoss(attn_kl=params["attn_kl"], kl_weight=params["kl_weight"], detach_targets=params["detach_targets"],)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                  factor=params["optim_factor"],
                                  patience=params["optim_step"],
                                  min_lr=1e-6,
                                  verbose=True)

    # Training
    best_val_score = 0.0
    save_dir = os.path.join(params["log_path"], "val_results")
    os.makedirs(save_dir, exist_ok=True)
    for epoch in range(params["num_epochs"]):
        train_stats, val_stats = {}, {}
        for fold in ['train', 'val']:
            print(f"*** Epoch {epoch + 1} {fold} fold ***")
            if fold == "train":
                model.train()
                for i, sample in tqdm(enumerate(train_loader), total=len(train_loader)):
                    images, targets = sample
                    images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                    outputs, attn, marginals = model(images)
                    if torch.isnan(outputs).any():
                        print("Oops")
                    losses = criterion(outputs, targets, marginals)
                    loss = losses[-1]
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    current_lr = optimizer.param_groups[0]['lr'] if scheduler is not None else params["lr"]
                    log_stats_classification(train_stats, outputs, targets, losses, batch_size=params["batch_size"],
                                         lr=current_lr)

            else:
                model.eval()
                with torch.no_grad():
                    for i, sample in tqdm(enumerate(val_loader), total=len(val_loader)):
                        images, targets = sample
                        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                        outputs, attn, marginals = model(images)
                        if torch.isnan(outputs).any():
                            print("Oops")
                        losses = criterion(outputs, targets, marginals)
                        current_lr = optimizer.param_groups[0]['lr'] if scheduler is not None else params["lr"]
                        log_stats_classification(val_stats, outputs, targets, losses, batch_size=params["batch_size"],
                                             lr=current_lr)
                val_loss, val_score = write_stats_classification(train_stats, val_stats, epoch,
                                                                 ret_metric=params["save_metric"])

        # progress LR scheduler
        if scheduler is not None:
            scheduler.step(val_loss)

        # Save parameters
        if val_score >= best_val_score and epoch >= params["min_epoch_save"]:
            model_file = os.path.join(log_dir,params["name"] + f'__best__epoch={epoch + 1:03d}_score={val_score:.4f}.pt')
            print(f'Model improved {params["save_metric"]} from {best_val_score:.4f} to {val_score:.4f}')
            print(f'Saving model at \'{model_file}\' ...')
            torch.save(model.state_dict(), model_file)
            best_val_score = val_score
            wandb.run.summary["best_val_score"] = best_val_score

        if params["chekpoint_save_interval"] > 0:
            if epoch % params["chekpoint_save_interval"] == 0 and epoch >= params["min_epoch_save"]:
                model_file = os.path.join(log_dir,
                                          params["name"] + f'__ckpt__epoch={epoch + 1:03d}_score={val_score:.4f}.pt')
                print(f"Saving model at '{model_file}' ...")
                torch.save(model.state_dict(), model_file)


if __name__ == "__main__":
    train()
    # for lamb in [0.1, 0.5]:
    # for lamb in [1e-4, 1e-3, 1e-2, 1e-1]:
    #     train(lamb)
    # models = 5
    # for i in range(5):
    #     print(f"********** Ensemble iteration {i+1:02d} **********")
    #     train(seed=i)
