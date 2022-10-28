import numpy as np

from tqdm import tqdm
from scripts.helpers import *

def mse_loss(y, tx, w):
    """Calculate the loss using Mean Squared Error (MSE).

    :param y: numpy array of shape (N,), N is the number of samples.
    :param tx: numpy array of shape (N,D), D is the number of features.
    :param w: numpy array of shape (D,), D is the number of features.
    :rtype: float
    :return: the mse loss value.
    """
    e = y - tx.dot(w)
    loss = 1/2 * np.mean(e**2)
    return loss


def mean_squared_error_gd(y, tx, initial_w=None, max_iters=100, gamma=0.25):
    """Gradient descent algorithm (GD) for linear regression.

    :param y: numpy array of shape (N,), N is the number of samples.
    :param tx: numpy array of shape (N,D), D is the number of features.
    :param initial_w: numpy array of shape (D,), D is the number of features.
    :param max_iters: scalar, the maximum number of iterations.
    :param gamma: scalar, the step size.
    :rtype: tuple
    :return: (w, loss), where w is the optimal weights, and loss is the loss value.
    """

    if max_iters == 0:
        loss = mse_loss(y, tx, initial_w)
        return initial_w, loss

    w = initial_w
    losses = []
    for _ in range(max_iters):
        e = y - tx.dot(w)
        grad = (-1/len(y)) * tx.T.dot(e)
        w = w - gamma * grad
        loss = mse_loss(y, tx, w)
        losses.append(loss)
    return w, losses[-1]


def mean_squared_error_sgd(
    y, tx, initial_w=None, max_iters=1000, gamma=1e-3, batch_size=1
):
    """Stochastic gradient descent algorithm (SGD) for linear regression.

    :param y: numpy array of shape (N,), N is the number of samples.
    :param tx: numpy array of shape (N,D), D is the number of features.
    :param initial_w: numpy array of shape (D,), D is the number of features.
    :param max_iters: scalar, the maximum number of iterations.
    :param gamma: scalar, the step size.
    :param batch_size: scalar, the number of samples in each batch.
    :rtype: tuple
    :return: (w, loss), where w is the optimal weights, and loss is the loss value.
    """
    losses = []
    w = initial_w

    for _ in tqdm(range(max_iters)):
        for y_batch, tx_batch in batch_iter(
            y, tx, batch_size=batch_size, num_batches=1, shuffle=True
        ):
            e = y_batch - tx_batch.dot(w)
            grad = (-1/len(y_batch)) * tx_batch.T.dot(e)
            w = w - gamma * grad
            loss = mse_loss(y_batch, tx_batch, w)
            losses.append(loss)
    return w, losses[-1]


def least_squares(y, tx):
    """Calculate the least squares solution.

    :param y: numpy array of shape (N,), N is the number of samples.
    :param tx: numpy array of shape (N,D), D is the number of features.
    :rtype: tuple
    :return: (w, loss), where w is the optimal weights, and loss is the loss value.
    """
    inv = np.linalg.inv(tx.T.dot(tx))
    w = (inv.dot(tx.T).dot(y))
    loss = mse_loss(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression algorithm.

    :param y: numpy array of shape (N,), N is the number of samples.
    :param tx: numpy array of shape (N,D), D is the number of features.
    :param lambda_: scalar, the regularization parameter.
    :rtype: tuple
    :return: (w, loss), where w is the optimal weights, and loss is the loss value.
    """
    N, D = tx.shape

    inv = np.linalg.inv(tx.T.dot(tx) + 2*N*lambda_ * np.identity(D))
    w = inv.dot(tx.T).dot(y)
    loss = mse_loss(y, tx, w)
    return w, loss


def logistic_regression(
    y, tx, initial_w=None, max_iters=225000, gamma=2e-3, batch_size=1, sgd=False
):
    """
    Logistic regression using GD or SGD.

    :param y: numpy array of shape (N,), N is the number of samples.
    :param tx: numpy array of shape (N,D), D is the number of features.
    :param initial_w: numpy array of shape (D,), D is the number of features.
    :param max_iters: scalar, the maximum number of iterations.
    :param gamma: scalar, the step size.
    :param batch_size: scalar, the number of samples in each batch.
    :param sgd: boolean, whether to use SGD or GD.
    :rtype: tuple
    :return: (w, loss), where w is the optimal weights, and loss is the loss value.
    """
    if max_iters == 0:
        logit = tx.dot(initial_w)
        loss = np.mean(np.log(1 + np.exp(logit)) - y * logit)
        return initial_w, loss

    losses = []
    w = initial_w

    if not sgd:
        for _ in range(max_iters):
            logit = tx.dot(w)
            gradient = tx.T.dot(sigmoid(logit) - y) / len(y)
            w = w - (gamma * gradient)
            logit = tx.dot(w)
            loss = np.mean(np.log(1 + np.exp(logit)) - y * logit)
            losses.append(loss)
    else:
        for _ in tqdm(range(max_iters)):
            for y_batch, tx_batch in batch_iter(
                y, tx, batch_size=batch_size, num_batches=1, shuffle=True
            ):
                logit = tx_batch.dot(w)
                loss = np.mean(np.log(1 + np.exp(logit)) - y_batch * logit)
                gradient = tx_batch.T.dot(
                    sigmoid(logit) - y_batch)
                w = w - (gamma * gradient)
                losses.append(loss)
    return w, losses[-1]


def reg_logistic_regression(
    y, tx, lambda_=1e-5, initial_w=None, max_iters=225000, gamma=2e-3, batch_size=1, sgd=False,
):
    """
    Regularized logistic regression using SGD or GD.

    :param y: numpy array of shape (N,), N is the number of samples.
    :param tx: numpy array of shape (N,D), D is the number of features.
    :param lambda_: scalar, the regularization parameter.
    :param initial_w: numpy array of shape (D,), D is the number of features.
    :param max_iters: scalar, the maximum number of iterations.
    :param gamma: scalar, the step size.
    :param batch_size: scalar, the number of samples in each batch.
    :param sgd: boolean, whether to use SGD or GD.
    :rtype: tuple
    :return: (w, loss), where w is the optimal weights, and loss is the loss value.
    """
    if max_iters == 0:
        logit = tx.dot(initial_w)
        loss = np.mean(np.log(1 + np.exp(logit)) - y * logit)
        return initial_w, loss

    losses = []
    w = initial_w

    if not sgd:
        for _ in range(max_iters):
            logit = tx.dot(w)
            gradient = (tx.T.dot(sigmoid(logit) - y) /
                        len(y)) + 2 * lambda_ * w
            w = w - (gamma * gradient)
            logit = tx.dot(w)
            loss = np.mean(np.log(1 + np.exp(logit)) - y *
                           logit)
            losses.append(loss)
    else:
        for _ in tqdm(range(max_iters)):
            for y_batch, tx_batch in batch_iter(
                y, tx, batch_size=batch_size, num_batches=1, shuffle=True
            ):
                logit = tx_batch.dot(w)
                loss = np.mean(np.log(1 + np.exp(logit)) - y_batch * logit)
                gradient = tx_batch.T.dot(
                    sigmoid(logit) - y_batch) + 2 * lambda_ * w
                w = w - (gamma * gradient)
                losses.append(loss)
    return w, losses[-1]
