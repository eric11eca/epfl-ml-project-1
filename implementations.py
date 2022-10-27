import numpy as np

from tqdm import tqdm
from scripts.helpers import *
from scripts.regression.loss import *
from scripts.regression.gradient import *


def mean_squared_error_gd(y, tx, initial_w=None, max_iters=100, gamma=0.25):
    """The Gradient Descent (GD) algorithm.
    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize (learning rate)
    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD
    """
    # Define parameters to store w and loss
    if max_iters == 0:
        loss = compute_loss(y, tx, initial_w)
        return initial_w, loss
    w = initial_w
    losses = []
    for n_iter in range(max_iters):
        e = y - tx.dot(w)
        grad = (-1/len(y)) * tx.T.dot(e)
        w = w - gamma * grad
        loss = compute_loss(y, tx, w)
        losses.append(loss)
    return w, losses[-1]


def mean_squared_error_sgd(
    y, tx, initial_w=None, max_iters=1000, gamma=1e-3, batch_size=1
):
    """The Stochastic Gradient Descent algorithm (SGD).
    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """
    losses = []
    w = initial_w
    for n_iter in tqdm(range(max_iters)):
        for y_batch, tx_batch in batch_iter(
            y, tx, batch_size=batch_size, num_batches=1, shuffle=True
        ):
            e = y_batch - tx_batch.dot(w)
            grad = (-1/len(y_batch)) * tx_batch.T.dot(e)
            w = w - gamma * grad
            e = y_batch - tx_batch.dot(w)
            loss = 1/2 * np.mean(e**2)
            losses.append(loss)

    return w, losses[-1]


def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        return_loss: whether return loss value

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.
    """
    inv = np.linalg.inv(tx.T.dot(tx))
    w = (inv.dot(tx.T).dot(y))
    loss = compute_loss(y, tx, w, opt='mse')
    return w, loss


def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar
        return_loss: bool

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])
    """
    N, D = tx.shape

    inv = np.linalg.inv(tx.T.dot(tx) + 2*N*lambda_ * np.identity(D))
    w = inv.dot(tx.T).dot(y)
    loss = compute_loss(y, tx, w, opt='mse')
    return w, loss


def logistic_regression(
    y, tx, initial_w=None, max_iters=225000, gamma=2e-3, batch_size=1, sgd=False
):
    """
    Logistic regression using GD or SGD.

    :param y: labels, size: (N,)
    :param tx: features, size: (N,D)
    :param initial_w: initial weight, size: (D,)
    :param max_iters: number of steps to run
    :param gamma: step size for (stochastic) gradient descent
    :return: (w, loss): last weight vector and loss, size: ((D,), 1)
    """
    if max_iters == 0:
        logit = tx.dot(initial_w)
        loss = np.mean(np.log(1 + np.exp(logit)) - y * logit)
        return initial_w, loss

    losses = []
    w = initial_w

    if not sgd:
        for n_iter in range(max_iters):
            logit = tx.dot(w)
            gradient = tx.T.dot(sigmoid(logit) - y) / len(y)
            w = w - (gamma * gradient)
            p_pred = sigmoid(logit)
            loss = -np.mean(y*np.log(p_pred)+(1-y)*np.log(p_pred))
            # loss = np.mean(np.log(1 + np.exp(-logit)) - y * logit)
            losses.append(loss)
    else:
        for n_iter in tqdm(range(max_iters)):
            for y_batch, tx_batch in batch_iter(
                y, tx, batch_size=batch_size, num_batches=1, shuffle=True
            ):
                logit = tx_batch.dot(w)
                p_pred = sigmoid(logit)
                # loss = -np.mean(y*np.log(p_pred)+(1-y)*np.log(p_pred))
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
    Regularized logistic regression using SGD.

    :param y: labels, size: (N,)
    :param tx: features, size: (N,D)
    :param lambda_: regularization coefficient
    :param initial_w: initial weight, size: (D,)
    :param max_iters: number of steps to run
    :param gamma: step size for (stochastic) gradient descent
    :return: (w, loss): last weight vector and loss, size: ((D,), 1)
    """
    if max_iters == 0:
        logit = tx.dot(initial_w)
        loss = np.mean(np.log(1 + np.exp(logit)) - y * logit)
        return initial_w, loss

    losses = []
    w = initial_w

    if not sgd:
        for n_iter in range(max_iters):
            logit = tx.dot(w)
            gradient = (tx.T.dot(sigmoid(logit) - y) /
                        len(y)) + 2 * lambda_ * w
            w = w - (gamma * gradient)
            logit = tx.dot(w)
            loss = np.mean(np.log(1 + np.exp(logit)) - y *logit)
            losses.append(loss)
    else:
        for n_iter in tqdm(range(max_iters)):
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
