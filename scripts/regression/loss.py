import numpy as np

from proj1_helpers import batch_iter


def mse_loss(e):
    """Calculate the MSE loss"""
    return 1/2 * np.mean(e**2)


def mae_loss(e):
    """Calculate the MAE loss"""
    return np.mean(np.abs(e))


def compute_loss(y, tx, w, opt="mse"):
    """
    Calculate the loss.
    You can calculate the loss using mse or mae.
    """
    e = y - tx.dot(w)
    if opt == "mse":
        return mse_loss(e)
    else:
        return mae_loss(e)


def log_likelihood_loss(y, tx, w):
    """
    Compute the log likelihood loss.
    """
    logit = tx.dot(w)
    loss = np.sum(np.log(1 + np.exp(logit)) - y * logit)
    return logit, loss
