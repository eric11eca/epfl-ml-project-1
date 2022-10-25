import logging

from implementations import *
from scripts.helpers import *
from scripts.dataset import *

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s:%(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger("runner")


class ModelAggregator:
    def __init__(self):
        self.aggregator = {
            "mse_gd": self.mean_square_error_gd,
            "mse_sgd": self.mean_square_error_sgd,
            'least_squares': self.least_squares,
            'ridge': self.ridge_regression,
            "logistic": self.logistic_regression,
            "reg_logistic": self.reg_logistic_regression,
        }

    def mean_square_error_gd(self, tx_train, y_train, tx_val, y_val, weights, params):
        max_iters = params["max_iters"]
        gamma = params["gamma"]
        logits, loss = mean_squared_error_gd(
            y=y_train,
            tx=tx_train,
            initial_w=weights,
            max_iters=max_iters,
            gamma=gamma
        )
        y_pred_train, train_loss = predict_binary(
            y_train,
            tx_train,
            w=logits,
            loss_type="mse"
        )
        y_pred_val, val_loss = predict_binary(
            y_val,
            tx_val,
            w=logits,
            loss_type="mse"
        )

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
            y=y_train,
            tx=tx_train,
            initial_w=weights,
            max_iters=max_iters,
            gamma=gamma,
            batch_size=batch_size
        )
        y_pred_train, train_loss = predict_binary(
            y_train,
            tx_train,
            w=logits,
            loss_type="mse"
        )
        y_pred_val, val_loss = predict_binary(
            y_val,
            tx_val,
            w=logits,
            loss_type="mse"
        )

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

    def ridge_regression(self, tx_train, y_train, tx_val, y_val, weights, params):
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
            y_train, tx_train, logits, loss_type="logistic"
        )
        y_pred_val, val_loss = predict_binary(
            y_val, tx_val, logits, loss_type="logistic"
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
            y_train, tx_train, logits, loss_type="logistic"
        )
        y_pred_val, val_loss = predict_binary(
            y_val, tx_val, logits, loss_type="logistic"
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
    if init_w is not None:
        weights = init_w
    else:
        weights = [np.zeros(tx.shape[1]) for _ in range(k_fold)]

    metric_log = {
        "train_loss": [],
        "val_loss": 1000 * np.ones(k_fold),
        "train_acc": [],
        "val_acc": np.zeros(k_fold),
        "val_f1": np.zeros(k_fold),
        "val_precision": np.zeros(k_fold),
        "val_recall": np.zeros(k_fold)
    }

    model_aggregator = ModelAggregator()
    best_weights = weights

    for k in range(k_fold):
        print(f"Cross validation: fold {k}")
        start = k * len(y) // k_fold
        end = (k + 1) * len(y) // k_fold

        tx_train = np.concatenate([tx[:start], tx[end:]])
        y_train = np.concatenate([y[:start], y[end:]])

        tx_val = tx[start:end]
        y_val = y[start:end]

        monitor = cv_params["monitor"]

        for epoch in range(cv_params["epochs"]):
            print(f"Epoch: {epoch}, num_steps: {cv_params['max_iters']}")
            output_dict = model_aggregator.aggregator[model](
                tx_train, y_train, tx_val, y_val,
                weights=weights[k], params=cv_params
            )

            train_acc, tr_p, tr_r, tr_f1 = compute_prf_binary(
                y_train, output_dict["y_pred_train"]
            )
            val_acc, va_p, va_r, va_f1 = compute_prf_binary(
                y_val, output_dict["y_pred_val"]
            )

            print(
                f"Epoch: {epoch}, val_acc: {val_acc}, val_loss: {output_dict['val_loss']}")
            weights[k] = output_dict["logits"]
            metric_log["train_loss"].append(output_dict["train_loss"])
            metric_log["train_acc"].append(train_acc)

            if val_acc > metric_log["val_acc"][k]:
                print("====================================================")
                print(f"Find best val_acc: {val_acc}. Best weights updated")
                best_weights[k] = weights[k]
                metric_log["val_acc"][k] = val_acc
                metric_log["val_f1"][k] = va_f1
                metric_log["val_precision"][k] = va_p
                metric_log["val_recall"][k] = va_r
                metric_log["val_loss"][k] = output_dict["val_loss"]

    for key in metric_log.keys():
        metric_log[key] = np.mean(metric_log[key])

    return weights, best_weights, metric_log


class Trainer:
    def __init__(
        self,
        tx,
        y,
        params=None,
        init_w=None,
        k_fold=10,
        model="reg_logistic",
        save_fig=False,
    ):
        self.y = y
        self.tx = tx
        self.params = params
        self.k_fold = k_fold
        self.model = model
        self.save_fig = save_fig
        self.monitor = params['monitor'][0]
        self.weights = [init_w for i in range(k_fold)]
        self.param_grid = build_parameter_grid(params)
        self.checkpoint = f"./log/{model}_{k_fold}fold_cv_best.json"

    def train(self):
        best_metric = 0.0
        best_params = {}

        for config in self.param_grid:
            print(f"Cross validation {self.model} with {config}")

            weights, best_weights, metric_log = cross_validation(
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
                write_json({
                    "best_params": best_params,
                    self.monitor: metric_log[self.monitor],
                    "k_best_weights": [w.tolist() for w in best_weights]
                }, self.checkpoint)

            print(f"current best params: {best_params}")

    def eval(self, tx_test, ids_test):
        print(f"Load best weights from {self.checkpoint}")
        checkpoint = read_json(self.checkpoint)
        best_weights = checkpoint["k_best_weights"]

        if "logistic" in self.model:
            model_type = "logistic"
        else:
            model_type = "linear"

        print(f"Test set evaluation")

        for k in range(self.k_fold):
            test_preds = predict_binary_test(
                tx=tx_test,
                w=best_weights[k],
                model_type=model_type
            )
            write_results_test(
                f"./output/{self.model}_fold_{k}_test_out.csv", ids_test, test_preds)


if __name__ == "__main__":
    logger.info("loading data ...")
    train_dataset = Dataset("./data", "train")
    test_dataset = Dataset("./data", "test")
    train_dataset.load_data()
    test_dataset.load_data()

    random_seed = 42
    np.random.seed(random_seed)

    ids = train_dataset.ids
    labels = train_dataset.labels
    features = train_dataset.data
    shuffle_idx = np.random.permutation(np.arange(len(labels)))
    shuffled_y = labels[shuffle_idx]
    shuffled_tx = features[shuffle_idx]
    init_w = np.random.uniform(low=-2.0, high=2.0, size=features.shape[1])
    model = "logistic"

    gd_gamma = [0.01, 0.05, 0.1, 0.25, 0.5]
    sgd_gamma = [5e-4, 1e-3, 2e-3, 5e-3, 0.01]
    k_fold = 4
    batch_size = 100
    warmup_propotion = (k_fold - 1) / k_fold

    hyper_params = {
        "batch_size": [batch_size],
        "monitor": ["val_acc"],
    }

    if model in ["mse_sgd", "logistic", "reg_logistic"]:
        split_rate = (k_fold - 1) / k_fold
        hyper_params["max_iters"] = [int(
            split_rate * len(shuffled_y) / batch_size)]
        hyper_params["epochs"] = [10]
        hyper_params["gamma"] = sgd_gamma
    elif model == "mse_gd":
        hyper_params["max_iters"] = [1]
        hyper_params["epochs"] = [200]
        hyper_params["gamma"] = gd_gamma
    else:
        hyper_params["epochs"] = [1]
        hyper_params["max_iters"] = [1]

    if model in ["ridge", "reg_logistic"]:
        hyper_params["lambda"] = [1e-6, 1e-5, 1e-4, 1e-3, 0.01]

    trainer = Trainer(
        tx=shuffled_tx,
        y=shuffled_y,
        params=hyper_params,
        init_w=init_w,
        k_fold=k_fold,
        model=model,
    )

    # trainer.train()

    test_features = test_dataset.data
    test_ids = test_dataset.ids

    trainer.eval(test_features, test_ids)
