import wandb
import numpy as np
from sklearn.metrics import mean_squared_error


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


def compute_y(stats):
    y_true, y_hat = zip(*stats['pred'])
    return y_true.numpy(), y_hat.numpy()


def write_stats_regression(train_stats, val_stats, epoch, ret_metric="rmse"):
    epoch = epoch + 1
    avg_train_loss = sum(train_stats['loss']) / len(train_stats['loss'])
    avg_val_loss = sum(val_stats['loss']) / len(val_stats['loss'])
    wandb.log({"avg_train_loss": avg_train_loss, "avg_val_loss": avg_val_loss, "epoch": epoch})
    if 'lr' in train_stats:
        wandb.log({"learning_rate": train_stats['lr'], "epoch": epoch})

    train_y_true, train_y_hat = compute_y(train_stats)
    val_y_true, val_y_hat = compute_y(val_stats)

    # RMSE
    wandb.log({"train_precision": mean_squared_error(train_y_true, train_y_hat, squared=False),
               "val_precision": mean_squared_error(val_y_true, val_y_hat, squared=False),
               "epoch": epoch})

    output = [avg_val_loss]
    if ret_metric.lower() == "rmse":
        output.append(mean_squared_error(val_y_true, val_y_hat, squared=False))
    else:
        raise NotImplementedError

    return tuple(output)
