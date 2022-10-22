# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np

from itertools import product


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",",
                      skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})


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


def build_parameter_grid(param_options):
    for item in product(*param_options.values()):
        yield dict(zip(param_options.keys(), item))


def predict_binary(y, tx, w, loss_type="logistic"):
    """
    :param y: ground truth labels, size: (N_Test,)
    :param tx: input features, size: (N_Test,D)
    :param w: trained weights
    :param loss_type: model type: ["mse", "rmse", "logistic"]
    :return: predict_y, test_loss
    """
    z = np.dot(tx, w)
    if loss_type == "mse":
        test_loss = 0.5 * np.mean((y - z)**2)
    elif loss_type == "rmse":
        test_loss = np.sqrt(np.mean((y - z)**2))
    elif loss_type == "logistic":
        test_loss = np.mean(np.log(1 + np.exp(z)) - y * z)
    else:
        raise ValueError("loss_type must be mse, rmse or logistic")

    if loss_type == "logistic":
        p_pred = np.exp(z) / (1 + np.exp(z))
    else:
        p_pred = z

    predict_y = list(map(lambda x: 0 if x < 0.5 else 1, p_pred))

    return predict_y, test_loss

def compute_prf_binary(label_y, predict_y):
    """
    :param label_y: ground truth labels
    :param predict_y: model predictions
    :return: precision, recall, f1
    """
    tp, fp, tn, fn = 1e-5, 1e-5, 1e-5, 1e-5
    for idx, label in enumerate(label_y):
        if label == 1:
            if predict_y[idx] == 1:
                tp += 1
            else:
                fn += 1
        else:
            if predict_y[idx] == 1:
                fp += 1
            else:
                tn += 1
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return accuracy, precision, recall, f1