import sys
import os
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

from config import TrainConfig
from dataio.dataloader import probe_data_folder, BraTS18
from train_utils import log_stats_regression, write_stats_regression
from models.resnet import get_resnet50_attn_regressor


def train(seed=None):
    raw_params = TrainConfig()
    # Serialize transforms
    # transforms = raw_params.config.pop("transforms")
    # raw_params.config["transforms"] = {"transforms": transforms.__str__()}

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

    train_metadata, val_metadata = probe_data_folder(params["data_path"],
                                                     train_frac=params["train_frac"],
                                                     bad_files=params["bad_files"])

    # Datasets
    train_dataset = BraTS18(params["data_path"], train_metadata) # TODO - transforms
    val_dataset = BraTS18(params["data_path"], val_metadata)

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
    model = get_resnet50_attn_regressor(**params)

    # Create log dir
    log_dir = os.path.join(params["log_path"], params["name"], datetime.now().strftime("%Y%m%d_%H:%M:%S"))
    os.makedirs(log_dir)
    print("Log dir: '{}' created".format(log_dir))
    pickle.dump(params, open(os.path.join(log_dir, "params.p"), "wb"))

    # CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = str(params["gpu_id"])
    device = torch.device("cuda" if torch.cuda.is_available() and params["use_gpu"] else "cpu")
    model = model.to(device)

    # Loss
    criterion = torch.nn.MSELoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                  factor=params["optim_factor"],
                                  patience=params["optim_step"],
                                  min_lr=1e-6,
                                  verbose=True)

    # Training
    best_val_score = 1e6
    save_dir = os.path.join(params["log_path"], "val_results")
    os.makedirs(save_dir, exist_ok=True)
    for epoch in range(params["num_epochs"]):
        train_stats, val_stats = {}, {}
        for fold in ['train', 'val']:
            print("*** Epoch {:04d} {} fold***".format(epoch + 1, fold))
            if fold == "train":
                model.train()
                for i, sample in tqdm(enumerate(train_loader), total=len(train_loader)):
                    images, targets = sample
                    images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                    outputs, _ = model(images)
                    if torch.isnan(outputs).any():
                        print("Oops")
                    loss = criterion(outputs, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    current_lr = optimizer.param_groups[0]['lr'] if scheduler is not None else params["lr"]
                    log_stats_regression(val_stats, outputs, targets, loss, batch_size=params["batch_size"],
                                         lr=current_lr)

            else:
                model.eval()
                with torch.no_grad():
                    for i, sample in tqdm(enumerate(val_loader), total=len(val_loader)):
                        images, targets = sample
                        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                        outputs, _ = model(images)
                        if torch.isnan(outputs).any():
                            print("Oops")
                        loss = criterion(outputs, targets)
                        current_lr = optimizer.param_groups[0]['lr'] if scheduler is not None else params["lr"]
                        log_stats_regression(val_stats, outputs, targets, loss, batch_size=params["batch_size"],
                                             lr=current_lr)
                val_loss, val_score = write_stats_regression(train_stats, val_stats, epoch,
                                                                 ret_metric=params["save_metric"])

        # progress LR scheduler
        if scheduler is not None:
            scheduler.step(val_loss)

        # Save parameters
        if val_score < best_val_score and epoch >= params["min_epoch_save"]:
            model_file = os.path.join(log_dir,params["name"] + '__best__epoch={:03d}_score={:.4f}.pt'.format(epoch + 1, val_score))
            print("Model improved '{}' from {:.4f} to {:.4f}".format(params["save_metric"], best_val_score, val_score))
            print("Saving model at '{}' ...".format(model_file))
            torch.save(model.state_dict(), model_file)
            best_val_score = val_score
            wandb.run.summary["best_val_score"] = best_val_score

        if params["chekpoint_save_interval"] > 0:
            if epoch % params["chekpoint_save_interval"] == 0 and epoch >= params["min_epoch_save"]:
                model_file = os.path.join(log_dir,
                                          params["name"] + '__ckpt__epoch={:03d}_score={:.4f}.pt'.format(epoch + 1, val_score))
                print("Saving model at '{}' ...".format(model_file))
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
