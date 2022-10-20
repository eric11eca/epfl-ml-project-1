import numpy as np

from scripts.proj1_helpers import batch_iter, predict_binary
from scripts.regression.loss import log_likelihood_loss
from scripts.regression.gradient import gradient_log_likelihood



"""
Gradient Descent
"""
import numpy as np

def compute_loss(y, tx, w):
    """Calculate the loss using either MSE or MAE.
    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.
    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss by MSE
    # ***************************************************
    pred = tx@w
    loss = np.mean((y-pred)**2, axis=0)
    return loss

def compute_gradient(y, tx, w):
    """Computes the gradient at w.
    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.
    Returns:
        An array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute gradient vector
    # ***************************************************
    grad = 2*np.mean(tx.T*(tx@w-y), axis=1)

    return grad 


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
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
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: compute gradient and loss
        # ***************************************************
        grad = compute_gradient(y, tx, initial_w)
        loss = compute_loss(y, tx, initial_w)
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: update w by gradient
        # ***************************************************
        w = w - gamma*grad

        # store w and loss
        ws.append(w)
        losses.append(loss)

    return ws, losses

"""
Stochastic Gradient Descent
"""


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
    


def mean_squared_error_sgd(y, tx, initial_w, batch_size, max_iters, gamma):
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

    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):

        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: implement stochastic gradient descent (n=1).
        # ***************************************************
        for y_, tx_ in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            grad = compute_gradient(y_, tx_, w)
            w = w - gamma*grad
            loss = compute_loss(y, tx, w)

            ws.append(w)
            losses.append(loss)

    return ws, losses





# linear regression using gradient descent
def least_squares_GD(y, tx, initial_w=None, max_iters=100, gamma=0.25):
    """
    :param y: labels, size: (N,)
    :param tx: features, size: (N,D)
    :param initial_w: initial weight, size: (D,)
    :param max_iters: number of steps to run
    :param gamma: step size for (stochastic) gradient descent
    :return: (w, loss): last weight vector and loss, size: ((D,), 1)
    """
    N = y.shape[0]
    losses = []
    w = initial_w if initial_w is not None else np.zeros(tx.shape[1])
    for mini_y, mini_tx in batch_iter(y, tx, batch_size=N, num_batches=max_iters, shuffle=True):
        loss = 0.5 * np.mean((mini_y - np.dot(mini_tx, w))**2)
        e = mini_y - np.dot(mini_tx, w)
        gradient = - np.dot(mini_tx.T, e) / N
        w = w - (gamma*gradient)
        losses.append(loss)
    return w, losses[-1]


# linear regression using stochastic gradient descent
def least_squares_SGD(y, tx, initial_w=None, max_iters=450000, gamma=1e-3):
    """
    :param y: labels, size: (N,)
    :param tx: features, size: (N,D)
    :param initial_w: initial weight, size: (D,)
    :param max_iters: number of steps to run
    :param gamma: step size for (stochastic) gradient descent
    :return: (w, loss): last weight vector and loss, size: ((D,), 1)
    """
    losses = []
    w = initial_w if initial_w is not None else np.zeros(tx.shape[1])
    for mini_y, mini_tx in batch_iter(y, tx, batch_size=1, num_batches=max_iters, shuffle=True):
        loss = 0.5 * np.mean((mini_y - np.dot(mini_tx, w))**2)
        e = mini_y - np.dot(mini_tx, w)
        gradient = - np.dot(mini_tx.T, e)
        w = w - (gamma*gradient)
        losses.append(loss)
    return w, losses[-1]


# least squares regression using normal equations
def least_squares(y, tx):
    """
    :param y: labels, size: (N,)
    :param tx: features, size: (N,D)
    :return: (w, loss): last weight vector and loss, size: ((D,), 1)
    """
    w = np.linalg.solve(np.dot(tx.T, tx), np.dot(tx.T, y))
    loss = 0.5 * np.mean((y - np.dot(tx, w))**2)

    return w, loss


# ridge regression using normal equations
def ridge_regression(y, tx, lambda_=1e-5):
    """
    :param y: labels, size: (N,)
    :param tx: features, size: (N,D)
    :param lambda_: regularization coefficient
    :return: (w, loss): last weight vector and loss, size: ((D,), 1)
    """
    N = y.shape[0]
    D = tx.shape[1]
    w = np.linalg.solve(np.dot(tx.T, tx) + 2 * N *
                        lambda_ * np.identity(D), np.dot(tx.T, y))
    loss = 0.5 * np.mean((y - np.dot(tx, w))**2)
    return w, loss


def logistic_regression(y, tx, initial_w=None, max_iters=225000, gamma=2e-3):
    """
    Logistic regression using SGD.

    :param y: labels, size: (N,)
    :param tx: features, size: (N,D)
    :param initial_w: initial weight, size: (D,)
    :param max_iters: number of steps to run
    :param gamma: step size for (stochastic) gradient descent
    :return: (w, loss): last weight vector and loss, size: ((D,), 1)
    """
    losses = []
    w = initial_w if initial_w is not None else np.zeros(tx.shape[1])
    for y_batch, tx_batch in batch_iter(
        y, tx, batch_size=1,
        num_batches=max_iters,
        shuffle=True
    ):
        logit, loss = log_likelihood_loss(y_batch, tx_batch, w)
        gradient = gradient_log_likelihood(
            y_batch, tx_batch, w, logit)
        w = w - (gamma * gradient)
        losses.append(loss)
    return w, losses[-1]


def reg_logistic_regression(y, tx, initial_w=None, lambda_=1e-5, max_iters=225000, gamma=2e-3):
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
    losses = []
    w = initial_w if initial_w is not None else np.zeros(tx.shape[1])
    for y_batch, batch_x in batch_iter(y, tx, batch_size=1, num_batches=max_iters, shuffle=True):
        logit, loss = log_likelihood_loss(y_batch, batch_x, w)
        gradient = np.dot(
            batch_x.T, (np.exp(logit) / (1 + np.exp(logit)) - y_batch)) + lambda_ * w
        w = w - (gamma * gradient)
        losses.append(loss)
    return w, losses[-1]


# regularized logistic regression with dynamic learning rate,
# combined with outlier filtering and feature augmentation
# def reg_logistic_dynamic(y, tx, y_valid, tx_valid, initial_w=None, max_epoch_iters=1000, gamma=2e-3, batch_size=1,
#                          lambda_=1e-5, dynamic_lr=True, k_cross=10, half_lr_count=2, early_stop_count=4):
#     """
#     :param y: labels, size: (N,)
#     :param tx: features, size: (N,D)
#     :param y_valid: validation set labels, size: (Nv,)
#     :param tx_valid: validation set features, size: (Nv,D)
#     :param initial_w: initial weight, size: (D,)
#     :param max_epoch_iters: number of epochs to run
#     :param gamma: step size (related to learning rate) for mini_batch gradient descent
#     :param batch_size: batch size for mini_batch gradient descent
#     :param lambda_: regularization coefficient
#     :param dynamic_lr: if use dynamic learning rate, otherwise fixed
#     :param k_cross: number of cross validation fold
#     :param half_lr_count: half the learning rate if validation loss increase in count of consecutive epochs
#     :param early_stop_count: stop training if validation loss increase in count of consecutive epochs
#     :return: (w, loss): last weight vector and loss, size: ((D,), 1)
#     """
#     data_size = len(y)
#     train_valid_ratio = (k_cross - 1) / k_cross
#     iters_per_epoch = int(data_size * train_valid_ratio / batch_size)
#     lr = gamma
#     loss = None
#     w = initial_w if initial_w is not None else np.zeros(tx.shape[1])

#     if dynamic_lr:
#         w_best = w.copy()
#         _, min_valid_loss = predict_binary(
#             y_valid, tx_valid, w, loss_type="logistic")
#         half_count = half_lr_count
#         stop_count = early_stop_count
#     else:
#         w_best, min_valid_loss, half_count, stop_count = None, None, None, None

#     for epoch in range(max_epoch_iters):
#         # perform SGD
#         for mini_y, mini_tx in batch_iter(y, tx, batch_size=batch_size, num_batches=iters_per_epoch, shuffle=True):
#             z = np.dot(mini_tx, w)
#             loss = np.mean(np.log(1 + np.exp(z)) - mini_y * z)
#             gradient = np.dot(
#                 mini_tx.T, (np.exp(z) / (1 + np.exp(z)) - mini_y)) + lambda_ * w
#             w = w - (lr * gradient)
#         # learning rate control
#         if dynamic_lr:
#             _, valid_loss = predict_binary(
#                 y_valid, tx_valid, w, loss_type="logistic")
#             print("Training Epoch: " + str(epoch) + ", Training Loss: " + str(loss) + ", Validation Loss: "
#                   + str(valid_loss) + ", Learning Rate:" + str(lr))
#             loss_drop = min_valid_loss - valid_loss
#             if valid_loss < min_valid_loss:
#                 min_valid_loss = valid_loss
#                 w_best = w.copy()
#             if loss_drop > 1e-5:
#                 half_count = half_lr_count
#                 stop_count = early_stop_count
#             else:
#                 half_count -= 1
#                 stop_count -= 1
#             if half_count == 0:
#                 lr = lr / 2.0
#                 half_count = half_lr_count
#             if stop_count == 0:
#                 break
#         else:
#             print("Training Epoch: " + str(epoch) +
#                   ", Training Loss: " + str(loss))

#     if dynamic_lr:
#         return w_best, loss
#     else:
#         return w, loss
