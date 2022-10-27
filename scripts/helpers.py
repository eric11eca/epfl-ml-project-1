# -*- coding: utf-8 -*-
"""some helper functions for project 1."""

import json
import csv
import numpy as np

from itertools import product


def read_file(path, mode="r", **kwargs):
    with open(path, mode=mode, **kwargs) as f:
        return f.read()


def write_file(data, path, mode="w", **kwargs):
    with open(path, mode=mode, **kwargs) as f:
        f.write(data)


def read_json(path, mode="r", **kwargs):
    return json.loads(read_file(path, mode=mode, **kwargs))


def write_json(data, path):
    return write_file(json.dumps(data, indent=2), path)


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
    """
    Build a grid of parameters to search over.

    :param param_options: dictionary of parameter names and lists of values to try
    :rtype: List[Dict]
    :return: list of Dict, each Dict is a set of parameters to try
    """
    for item in product(*param_options.values()):
        yield dict(zip(param_options.keys(), item))


def predict_val(y, tx, w, loss_type="logistic"):
    """
    Prediction function for validation data, where labels are  either 0 or 1

    :param y: ground truth labels, size: (N_Test,)
    :param tx: input features, size: (N_Test,D)
    :param w: trained weights
    :param loss_type: types of loss used
    :rtype: List[int], float
    :return: y_preds, val_loss
    """
    logits = tx.dot(w)

    if loss_type == "mse":
        val_loss = 1/2 * np.mean((y - logits)**2)
    elif loss_type == "logistic":
        print(y)
        val_loss = np.mean(np.log(1 + np.exp(logits)) - y * logits)

    preds = sigmoid(logits) if loss_type == "logistic" else logits
    y_preds = list(map(lambda x: 0 if x < 0.5 else 1, preds))

    return y_preds, val_loss


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


def predict_test(tx, w, logistic=False):
    """
    Prediction function for test data, where labels are  either -1 or 1

    :param tx: input features, size: (N_Test,D)
    :param w: trained weights
    :param logistic: whether to use logistic regression
    :rtype: List[int]
    :return: y_preds
    """
    logits = np.dot(tx, w)
    preds = sigmoid(logits) if logistic else logits
    y_preds = list(map(lambda x: -1 if x < 0.5 else 1, preds))
    return y_preds


def horizontal_voting(fold, model):
    """
    Horizontal voting for the test data

    :param fold: fold number for corss validation
    :param model: model used for learning
    :rtype: List[int], List[int]
    :return: ids, y_preds_vote
    """

    all_y_preds = []
    for i in range(fold):
        ids, y_preds = read_test_results(
            f"./output/{model}_fold_{i}_test_out.csv")
        all_y_preds.append(y_preds)

    all_y_preds = np.array(all_y_preds)

    pos = np.count_nonzero(all_y_preds == 1, axis=0)
    neg = np.count_nonzero(all_y_preds == -1, axis=0)

    y_preds_vote = []
    for pos_vote, neg_vote in zip(pos, neg):
        if pos_vote >= neg_vote:
            y_preds_vote.append(1)
        else:
            y_preds_vote.append(-1)

    return ids, y_preds_vote


def read_test_results(file_path):
    ids = []
    y_preds = []
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            ids.append(int(row['Id']))
            y_preds.append(int(row['Prediction']))
    return ids, y_preds


def sigmoid(t):
    """
    apply sigmoid function on t.

    :param t: A scalar or numpy array.
    :rtype: numpy.ndarray
    :return: sigmoid(t)
    """
    return 1 / (1 + np.exp(-t))


def learning_rate_schedular(learning_rate, epoch, decay_rate=0.9):
    """
    Learning rate schedular

    :param learning_rate: initial learning rate
    :param epoch: current epoch
    :param decay_rate: decay rate
    :rtype: float
    :return: new learning rate
    """
    return learning_rate * (decay_rate ** epoch)
