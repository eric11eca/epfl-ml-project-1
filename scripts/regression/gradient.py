import numpy as np


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute gradient and error vector
    # ***************************************************

    e = y - tx.dot(w)
    delta = -1/len(y) * tx.T.dot(e)
    return delta


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: implement stochastic gradient computation.It's same as the gradient descent.
    # ***************************************************

    e = y - tx.dot(w)
    delta = -1/len(y) * tx.T.dot(e)
    return delta


def compute_subgradient_mae(y, tx, w):
    """Compute a subgradient of the MAE at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        A numpy array of shape (2, ) (same shape as w), containing the subgradient of the MAE at w.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute subgradient gradient vector for MAE
    # ***************************************************

    N = len(y)
    e = y - tx.dot(w)
    return -1/N * tx.T.dot(np.sign(e)), e


def gradient_log_likelihood(y, tx, logit):
    """
    Compute the gradient of the log likelihood loss.
    """
    return tx.T.dot(np.exp(logit) / (1 + np.exp(logit)) - y)
