import wandb
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, confusion_matrix, precision_score, recall_score, roc_auc_score, \
    f1_score, accuracy_score, balanced_accuracy_score


def inverse_class_weights(class_counts):
    class_prop = class_counts / sum(class_counts)
    class_prop_inverse = 1.0 / class_prop
    return torch.tensor(class_prop_inverse / sum(class_prop_inverse), dtype=torch.float)


def log_stats_regression(stats, outputs, targets, loss, batch_size=1, lr=None):
    loss = loss.item() / batch_size
    if 'loss' in stats:
        stats['loss'].append(loss)
    else:
        stats['loss'] = [loss]


    np_outputs = outputs.detach().cpu().numpy().tolist()
    np_targets = targets.detach().cpu().numpy().tolist()
    preds = [(y, p) for (y, p) in zip(np_targets, np_outputs)]
    if 'pred' in stats:
        stats['pred'] += preds
    else:
        stats['pred'] = preds
    if lr:
        stats['lr'] = lr


def write_stats_regression(train_stats, val_stats, epoch, ret_metric="rmse"):
    epoch = epoch + 1
    avg_train_loss = sum(train_stats['loss']) / len(train_stats['loss'])
    avg_val_loss = sum(val_stats['loss']) / len(val_stats['loss'])
    wandb.log({"avg_train_loss": avg_train_loss, "avg_val_loss": avg_val_loss, "epoch": epoch})
    if 'lr' in train_stats:
        wandb.log({"learning_rate": train_stats['lr'], "epoch": epoch})

    train_y_true, train_y_hat = zip(*train_stats['pred'])
    val_y_true, val_y_hat = zip(*val_stats['pred'])

    # RMSE
    wandb.log({"train_rmse": mean_squared_error(train_y_true, train_y_hat, squared=False),
               "val_rmse": mean_squared_error(val_y_true, val_y_hat, squared=False),
               "epoch": epoch})

    output = [avg_val_loss]
    if ret_metric.lower() == "rmse":
        output.append(mean_squared_error(val_y_true, val_y_hat, squared=False))
    else:
        raise NotImplementedError

    return tuple(output)


def compute_y(stats):
    # Compute y_hat / y_pred
    y_true, y_hat = zip(*stats['pred'])
    y_true, y_hat = torch.tensor(y_true, dtype=torch.float32), torch.tensor(y_hat, dtype=torch.float32)
    if any(y_hat.sum(dim=1) != 1.0):
        y_hat = torch.softmax(y_hat, dim=1)
    y_hat_max, y_pred = torch.max(y_hat, dim=1)
    y_pred = y_pred.float()

    y_true = y_true.numpy()
    y_hat = y_hat.numpy()
    y_hat_max = y_hat_max.numpy()
    y_pred = y_pred.numpy()

    return y_true, y_hat, y_hat_max, y_pred


def write_stats_classification(train_stats, val_stats, epoch, ret_metric="balanced_accuracy_score"):
    epoch = epoch + 1
    for key, value in train_stats.items():
        if "loss" in key:
            avg_train_loss = sum(train_stats[key]) / len(train_stats[key])
            avg_val_loss = sum(val_stats[key]) / len(val_stats[key])
            wandb.log({f"train_{key}": avg_train_loss, f"val_{key}": avg_val_loss, "epoch": epoch})
    avg_val_total_loss = sum(val_stats['total_loss']) / len(val_stats['total_loss'])
    if 'lr' in train_stats:
        wandb.log({"learning_rate": train_stats['lr'], "epoch": epoch})
    if 'lamb' in train_stats:
        wandb.log({"entropy_lambda": train_stats['lamb'], "epoch": epoch})

    train_y_true, train_y_hat, train_y_hat_max, train_y_pred = compute_y(train_stats)
    val_y_true, val_y_hat, val_y_hat_max, val_y_pred = compute_y(val_stats)

    # Precision
    wandb.log({"train_precision": precision_score(train_y_true, train_y_pred, average="macro", zero_division=0),
               "val_precision": precision_score(val_y_true, val_y_pred, average="macro", zero_division=0),
               "epoch": epoch})

    # Recall
    wandb.log({"train_recall": recall_score(train_y_true, train_y_pred, average="macro", zero_division=0),
               "val_recall": recall_score(val_y_true, val_y_pred, average="macro", zero_division=0),
               "epoch": epoch})

    # Balanced Accuracy
    wandb.log({"train_balanced_accuracy": balanced_accuracy_score(train_y_true, train_y_pred),
               "val_balanced_accuracy": balanced_accuracy_score(val_y_true, val_y_pred),
               "epoch": epoch})

    # Macro_F1_Score
    wandb.log({"train_macro_f1_score": f1_score(train_y_true, train_y_pred, average="macro", zero_division=0),
               "val_macro_f1_score": f1_score(val_y_true, val_y_pred, average="macro", zero_division=0),
               "epoch": epoch})

    # # AUC_ROC_ovo
    # wandb.log({"train_auc_roc_ovo": roc_auc_score(train_y_true, train_y_hat, average="macro", multi_class="ovo"),
    #            "val_auc_roc_ovo": roc_auc_score(val_y_true, val_y_hat, average="macro", multi_class="ovo"),
    #            "epoch": epoch})

    output = [avg_val_total_loss]
    if ret_metric.lower() == "balanced_accuracy_score":
        output.append(balanced_accuracy_score(val_y_true, val_y_pred))
    elif ret_metric.lower() == "macro_f1_score":
        output.append(f1_score(val_y_true, val_y_pred, average="macro", zero_division=0))
    # elif ret_metric.lower() == "auc_roc_ovo":
    #     output.append(roc_auc_score(val_y_true, val_y_hat, average="macro", multi_class="ovo"))
    else:
        raise NotImplementedError

    return tuple(output)


def log_stats_classification(stats, outputs, targets, losses, batch_size=None, lr=None):
    lamb, kl_losses = None, []
    if len(losses) == 1:
        ce_loss = total_loss = losses[0]
    elif len(losses) == 3:
        ce_loss, kl_losses, total_loss = losses
        kl_losses = [k.detach() for k in kl_losses]
    else:
        raise ValueError(f"Cannot unpack losses: {losses}")
    ce_loss = ce_loss.item() / batch_size
    if 'cross_entropy_loss' in stats:
        stats['cross_entropy_loss'].append(ce_loss)
    else:
        stats['cross_entropy_loss'] = [ce_loss]
    for i, kl_loss in enumerate(kl_losses):
        if kl_loss is not None:
            kl_loss = kl_loss.item() / batch_size
            if f'kl_loss{i+1}' in stats:
                stats[f'kl_loss{i+1}'].append(kl_loss)
            else:
                stats[f'kl_loss{i+1}'] = [kl_loss]
    if lamb is not None:
        stats['lamb'] = lamb
    else:
        total_loss = losses[0]

    total_loss = total_loss.item() / batch_size
    if 'total_loss' in stats:
        stats['total_loss'].append(total_loss)
    else:
        stats['total_loss'] = [total_loss]

    np_outputs = outputs.detach().cpu().numpy().tolist()
    np_targets = targets.detach().cpu().numpy().tolist()
    preds = [(y, p) for (y, p) in zip(np_targets, np_outputs)]
    if 'pred' in stats:
        stats['pred'] += preds
    else:
        stats['pred'] = preds
    if lr:
        stats['lr'] = lr