import logging
import argparse

from pprint import pprint
from implementations import *
from scripts.helpers import *
from scripts.dataset import *

logging.basicConfig(
    filename="file.log",
    level=logging.DEBUG,
    format="%(asctime)s:%(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger("runner")


class ModelAggregator:
    def __init__(self):
        self.aggregator = {
            "mean_square_error_gd": self.mean_square_error_gd,
            "mean_square_error_sgd": self.mean_square_error_sgd,
            'least_squares': self.least_squares,
            'ridge_regression': self.ridge_regression,
            "logistic_regression": self.logistic_regression,
            "reg_logistic_regression": self.reg_logistic_regression,
        }

    def mean_square_error_gd(self, tx_train, y_train, tx_val, y_val, weights, params):
        max_iters = params["max_iters"]
        gamma = params["gamma"]
        logits, loss = mean_squared_error_gd(
            y_train, tx_train, weights, max_iters, gamma
        )
        y_pred_train, train_loss = predict_binary(
            y_train, tx_train, logits, loss_type="mse"
        )
        y_pred_val, val_loss = predict_binary(
            y_val, tx_val, logits, loss_type="mse")

        return {
            "logits": logits,
            "loss": loss,
            "y_pred_train": y_pred_train,
            "train_loss": train_loss,
            "y_pred_val": y_pred_val,
            "val_loss": val_loss,
        }

    def mean_square_error_sgd(self, tx_train, y_train, tx_val, y_val, weights, params):
        max_iters = params["max_iters"]
        gamma = params["gamma"]
        batch_size = params["batch_size"]
        logits, loss = mean_squared_error_sgd(
            y_train, tx_train, weights, batch_size, max_iters, gamma
        )
        y_pred_train, train_loss = predict_binary(
            y_train, tx_train, weights, loss_type="mse"
        )
        y_pred_val, val_loss = predict_binary(
            y_val, tx_val, weights, loss_type="mse")

        return {
            "logits": logits,
            "loss": loss,
            "y_pred_train": y_pred_train,
            "train_loss": train_loss,
            "y_pred_val": y_pred_val,
            "val_loss": val_loss,
        }

    def least_squares(self, tx_train, y_train, tx_val, y_val, weights, params):
        logits, loss = least_squares(y_train, tx_train)
        y_pred_train, train_loss = predict_binary(
            y_train, tx_train, logits, loss_type="rmse")
        y_pred_val, val_loss = predict_binary(
            y_val, tx_val, logits, loss_type="rmse")
        return{
            'logits': logits,
            'loss': loss,
            'y_pred_train': y_pred_train,
            'train_loss': train_loss,
            'y_pred_val': y_pred_val,
            'val_loss': val_loss
        }

    def ridge_regression(self, tx_train, y_train, tx_val, y_val, params):
        lambda_ = params['lambda']
        logits, loss = ridge_regression(y_train, tx_train, lambda_)
        y_pred_train, train_loss = predict_binary(
            y_train, tx_train, logits, loss_type="rmse")
        y_pred_val, val_loss = predict_binary(
            y_val, tx_val, logits, loss_type="rmse")
        return{
            'logits': logits,
            'loss': loss,
            'y_pred_train': y_pred_train,
            'train_loss': train_loss,
            'y_pred_val': y_pred_val,
            'val_loss': val_loss
        }

    def logistic_regression(self, tx_train, y_train, tx_val, y_val, weights, params):
        max_iters = params["max_iters"]
        gamma = params["gamma"]
        logits, loss = logistic_regression(
            y_train, tx_train, weights, max_iters, gamma)
        y_pred_train, train_loss = predict_binary(
            y_train, tx_train, weights, loss_type="logistic"
        )
        y_pred_val, val_loss = predict_binary(
            y_val, tx_val, weights, loss_type="logistic"
        )

        return {
            "logits": logits,
            "loss": loss,
            "y_pred_train": y_pred_train,
            "train_loss": train_loss,
            "y_pred_val": y_pred_val,
            "val_loss": val_loss,
        }

    def reg_logistic_regression(
        self, tx_train, y_train, tx_val, y_val, weights, params
    ):
        max_iters = params["max_iters"]
        gamma = params["gamma"]
        lambda_ = params["lambda"]
        logits, loss = reg_logistic_regression(
            y_train, tx_train, weights, lambda_, max_iters, gamma
        )
        y_pred_train, train_loss = predict_binary(
            y_train, tx_train, weights, loss_type="logistic"
        )
        y_pred_val, val_loss = predict_binary(
            y_val, tx_val, weights, loss_type="logistic"
        )

        return {
            "logits": logits,
            "loss": loss,
            "y_pred_train": y_pred_train,
            "train_loss": train_loss,
            "y_pred_val": y_pred_val,
            "val_loss": val_loss,
        }

    def train(self, method, tx, y, weights, params):
        return self.aggregator[method](tx, y, weights, params)


def cross_validation(
    tx, y, init_w=None, k_fold=10, cv_params=None, model="logistic_regression"
):
    """
    Cross validation for the given model
    """

    metric_log = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_f1": [],
        "val_f1": [],
        "train_precision": [],
        "val_precision": [],
        "train_recall": [],
        "val_recall": [],
    }

    if init_w is not None:
        weights = init_w
    else:
        weights = [np.zeros(tx.shape[1]) for _ in range(k_fold)]

    model_aggregator = ModelAggregator()

    for k in range(k_fold):
        print(f"Cross validation: fold {k}")
        start = k * len(y) // k_fold
        end = (k + 1) * len(y) // k_fold

        tx_train = np.concatenate([tx[:start], tx[end:]])
        y_train = np.concatenate([y[:start], y[end:]])

        tx_val = tx[start:end]
        y_val = y[start:end]

        if model in ['least_squares', 'ridge_regression']:
            output_dict = model_aggregator.aggregator[model](
                tx_train, y_train, tx_val, y_val, params=cv_params
            )
        else:
            output_dict = model_aggregator.aggregator[model](
                tx_train, y_train, tx_val, y_val, weights=weights[k], params=cv_params
            )

        train_acc, tr_p, tr_r, tr_f1 = compute_prf_binary(
            y_train, output_dict["y_pred_train"]
        )
        val_acc, va_p, va_r, va_f1 = compute_prf_binary(
            y_val, output_dict["y_pred_val"]
        )

        metric_log["train_loss"].append(output_dict["train_loss"])
        metric_log["val_loss"].append(output_dict["val_loss"])
        metric_log["train_acc"].append(train_acc)
        metric_log["train_precision"].append(tr_p)
        metric_log["train_recall"].append(tr_r)
        metric_log["train_f1"].append(tr_f1)
        metric_log["val_acc"].append(val_acc)
        metric_log["val_precision"].append(va_p)
        metric_log["val_recall"].append(va_r)
        metric_log["val_f1"].append(va_f1)

    for key in metric_log.keys():
        metric_log[key] = np.mean(metric_log[key])

    return weights, metric_log


class Trainer:
    def __init__(
        self,
        tx,
        y,
        params=None,
        init_w=None,
        k_fold=10,
        model="logistic_regression",
        monitor="val_acc",
    ):
        self.y = y
        self.tx = tx
        self.params = params
        self.k_fold = k_fold
        self.init_w = init_w
        self.model = model
        self.monitor = monitor
        self.warmup_propotion = (k_fold - 1) / k_fold
        self.weights = [init_w for i in range(k_fold)]
        self.param_grid = build_parameter_grid(params)

        file_path = f"./log/{model}_k{k_fold}.jsonl"

    def train_with_loop(self):
        best_metric = 0.0
        best_params = {}

        for config in self.param_grid:
            # num_epoch = config["num_epoch"]

            print(f"Cross validation {self.model} with {config}")

            # for epoch in range(num_epoch):
            weights, metric_log = cross_validation(
                tx=self.tx,
                y=self.y,
                init_w=self.weights,
                k_fold=self.k_fold,
                cv_params=config,
                model=self.model,
            )

            print(f"Finish Validation: {metric_log}")

            if metric_log[self.monitor] > best_metric:
                print(
                    f"New best {self.monitor} found: {metric_log[self.monitor]}")
                best_metric = metric_log[self.monitor]
                best_params = config

            print(f"current best params: {best_params}")

            #  save best parameters

    def train_without_loop(self):
        best_metric = 0.0
        best_params = {}

        for config in self.param_grid:
            weights, metric_log = cross_validation(
                tx=self.tx,
                y=self.y,
                init_w=self.init_w,
                k_fold=self.k_fold,
                cv_params=config,
                model=self.model,
            )

            if metric_log[self.monitor] > best_metric:
                print(
                    f"New best {self.monitor} found: {metric_log[self.monitor]}")
                best_metric = metric_log[self.monitor]
                best_params = config

                #  save best parameters


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--datadir', type=str, default="./data", required=False)
    parser.add_argument('--model', type=str, default="logistic_regression", required=False)
    # Parse the argument
    args = parser.parse_args()

    print("loading data ...")
    train_dataset = Dataset(args.datadir, "train")
    train_dataset.load_data()

    random_seed = 42
    np.random.seed(random_seed)

    labels = train_dataset.labels
    features = train_dataset.data
    init_w = np.random.uniform(low=-2.0, high=2.0, size=features.shape[1])

    gd_gamma = [0.01, 0.05, 0.1, 0.25, 0.5]
    sgd_gamma = [2e-3]  # [5e-4, 1e-3, 2e-3, 5e-3, 0.01]
    epochs_GD = 100
    epochs_SGD = 10
    k_fold = 10
    batch_size = 64
    warmup_propotion = (k_fold - 1) / k_fold
    max_iters = 100

    hyper_params = {
        "gamma": sgd_gamma,
        "lambda": [1e-6, 1e-5, 1e-4, 1e-3, 0.01],
        "num_epoch": [epochs_SGD],
        "max_iters": [max_iters],
        "batch_size": [batch_size],
    }

    shuffle_idx = np.random.permutation(np.arange(len(labels)))
    shuffled_y = labels[shuffle_idx]
    shuffled_tx = features[shuffle_idx]

    if args.model in ['least_squares', 'ridge_regression']:
        trainer = Trainer(
            tx=shuffled_tx,
            y=shuffled_y,
            params=hyper_params,
            init_w=init_w,
            k_fold=k_fold,
            model=args.model,
        )
        trainer.train_without_loop()
    else:
        trainer = Trainer(
            tx=shuffled_tx,
            y=shuffled_y,
            params=hyper_params,
            init_w=init_w,
            k_fold=k_fold,
            model=args.model,
        )
        trainer.train_with_loop()
