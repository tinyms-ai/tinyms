import math
import tinyms as ts


def cyclegan_lr(max_epoch, n_epoch, dataset_size):
    """
    Generate learning rate for cycle_gan.

    Args:
       max_epoch (int): Epoch size for training.
       n_epoch (int): Number of epochs with the initial learning rate.
       dataset_size (int): Total size of dataset.

    Returns:
       Tensor, learning rate.
    """
    n_epochs_decay = max_epoch - n_epoch
    lrs = [0.0002] * dataset_size * n_epoch
    lr_epoch = 0
    for epoch in range(n_epochs_decay):
        lr_epoch = 0.0002 * (n_epochs_decay - epoch) / n_epochs_decay
        lrs += [lr_epoch] * dataset_size
    lrs += [lr_epoch] * dataset_size * (max_epoch - n_epochs_decay - n_epoch)
    return ts.array(lrs, dtype=ts.float32)
