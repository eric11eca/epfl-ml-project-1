# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
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
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})


# training data batch generator
def batch_iter(y, tx, batch_size, num_batches=1, seed=7, shuffle=True):
    """
    :param y: labels, size: (N,)
    :param tx: features, size: (N,D)
    :param batch_size: size of mini_batch B
    :param num_batches: number of batches generated K
    :param seed: random seed for shuffle
    :param shuffle: if shuffle the data
    :return: mini-batch data: ((y_mini_batch_1, tx_mini_batch_1), ..., (y_mini_batch_K, tx_mini_batch_K))
    """
    data_size = y.shape[0]
    np.random.seed(seed)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx

    for batch_num in range(int(num_batches)):
        start_index = int(batch_num * batch_size) % data_size
        end_index = int((batch_num + 1) * batch_size) % data_size
        if start_index < end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
        else:
            if shuffle:
                new_shuffle_indices = np.random.permutation(
                    np.arange(data_size))
                new_shuffled_y = y[new_shuffle_indices]
                new_shuffled_tx = tx[new_shuffle_indices]
            else:
                new_shuffled_y = y
                new_shuffled_tx = tx
            yield np.r_[shuffled_y[start_index:], new_shuffled_y[:end_index]], np.r_[shuffled_tx[start_index:],
                                                                                     new_shuffled_tx[:end_index]]
            shuffled_y = new_shuffled_y
            shuffled_tx = new_shuffled_tx
