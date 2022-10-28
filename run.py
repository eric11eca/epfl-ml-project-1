import os
import argparse

from implementations import *
from scripts.helpers import *
from scripts.dataset import *
from scripts.visualization import *


class ModelAggregator:
    """Model wrapper for computing an epoch of the model"""
    def __init__(self):
        self.aggregator = {
            "mse_gd": self.mean_square_error_gd,
            "mse_sgd": self.mean_square_error_sgd,
            'least_squares': self.least_squares,
            'ridge': self.ridge_regression,
            "logistic": self.logistic_regression,
            "reg_logistic": self.reg_logistic_regression,
            "logistic_w_outlier": self.logistic_regression,
            "reg_logistic_w_outlier": self.reg_logistic_regression,
        }

    def mean_square_error_gd(self, tx_train, y_train, tx_val, y_val, weights, params):
        """Wrapper for mean square error gradient descent"""
        max_iters = params["max_iters"]
        gamma = params["gamma"]
        logits, loss = mean_squared_error_gd(
            y=y_train,
            tx=tx_train,
            initial_w=weights,
            max_iters=max_iters,
            gamma=gamma
        )
        y_pred_train, train_loss = predict_val(
            y_train,
            tx_train,
            w=logits,
            loss_type="mse"
        )
        y_pred_val, val_loss = predict_val(
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
        """Wrapper for mean square error stochastic gradient descent"""
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
        y_pred_train, train_loss = predict_val(
            y_train,
            tx_train,
            w=logits,
            loss_type="mse"
        )
        y_pred_val, val_loss = predict_val(
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
        """Wrapper for least squares"""
        logits, loss = least_squares(y_train, tx_train)
        y_pred_train, train_loss = predict_val(
            y_train, tx_train, logits, loss_type="rmse")
        y_pred_val, val_loss = predict_val(
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
        """Wrapper for ridge regression"""
        lambda_ = params['lambda']
        logits, loss = ridge_regression(y_train, tx_train, lambda_)
        y_pred_train, train_loss = predict_val(
            y_train, tx_train, logits, loss_type="rmse")
        y_pred_val, val_loss = predict_val(
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
        """Wrapper for logistic regression"""
        max_iters = params["max_iters"]
        gamma = params["gamma"]
        logits, loss = logistic_regression(
            y_train, tx_train, weights, max_iters, gamma, sgd=True)
        y_pred_train, train_loss = predict_val(
            y_train, tx_train, logits, loss_type="logistic"
        )
        y_pred_val, val_loss = predict_val(
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
        """Wrapper for regularized logistic regression"""
        max_iters = params["max_iters"]
        gamma = params["gamma"]
        lambda_ = params["lambda"]
        logits, loss = reg_logistic_regression(
            y_train, tx_train, lambda_, weights, max_iters, gamma, sgd=True
        )

        y_pred_train, train_loss = predict_val(
            y_train, tx_train, logits, loss_type="logistic"
        )
        y_pred_val, val_loss = predict_val(
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
        """Unified training wrapper function"""
        return self.aggregator[method](tx, y, weights, params)


def cross_validation(
    tx, y,
    init_w=None,
    k_fold=10,
    cv_params=None,
    model="logistix",
    save_fig=False,
    degree=4,
    feature_names=None,
    lr_decay=False,
):
    """
    Cross validation for the given model

    :param tx: training data
    :param y: training labels
    :param init_w: initial weights
    :param k_fold: number of folds
    :param cv_params: parameters for the cross validation
    :param model: model to use
    :param save_fig: save the figure
    :param degree: degree of the polynomial expansion
    :param feature_names: names of the features
    :param lr_decay: use learning rate decay
    :rtype: list, list, dict
    :return: k final weights, k best weights, metric logs
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
    training_tracker = {}

    for k in range(k_fold):
        print(f"Cross validation: fold {k}")
        start = k * len(y) // k_fold
        end = (k + 1) * len(y) // k_fold

        tx_train = np.concatenate([tx[:start], tx[end:]])
        y_train = np.concatenate([y[:start], y[end:]])

        tx_val = tx[start:end]
        y_val = y[start:end]

        training_tracker[f'fold-{k}'] = {k: [] for k in metric_log.keys()}

        if lr_decay:
            lr_schedular = LearningRateScheduler(
                epochs=cv_params['epochs'], 
                initial_learning_rate=cv_params['gamma'],
                schedule=cv_params["lr_schedule"]
            )

        for epoch in range(cv_params["epochs"]):
            print(f"Epoch: {epoch}, num_steps: {cv_params['max_iters']}")
            output_dict = model_aggregator.aggregator[model](
                tx_train, y_train, tx_val, y_val,
                weights=weights[k], params=cv_params
            )

            train_acc, _, _, _ = compute_metrics(
                y_train, output_dict["y_pred_train"]
            )
            val_acc, val_p, val_r, val_f1 = compute_metrics(
                y_val, output_dict["y_pred_val"]
            )

            print(
                f"Epoch: {epoch}, val_acc: {val_acc}, val_loss: {output_dict['val_loss']}")
            weights[k] = output_dict["logits"]
            # metric_log["train_loss"].append(output_dict["train_loss"])
            # metric_log["train_acc"].append(train_acc)

            training_tracker[f'fold-{k}']["train_loss"].append(
                output_dict["train_loss"])
            training_tracker[f'fold-{k}']["train_acc"].append(train_acc)
            training_tracker[f'fold-{k}']["val_loss"].append(
                output_dict["val_loss"])
            training_tracker[f'fold-{k}']["val_acc"].append(val_acc)
            training_tracker[f'fold-{k}']["val_f1"].append(val_f1)
            training_tracker[f'fold-{k}']["val_precision"].append(val_p)
            training_tracker[f'fold-{k}']["val_recall"].append(val_r)

            if lr_decay:
                cv_params['gamma'] = lr_schedular.get_learning_rate(epoch)
                print(f"Decaying learning rate to: {cv_params['gamma']}")

            if val_acc > metric_log["val_acc"][k]:
                print("====================================================")
                print(f"Find best val_acc: {val_acc}. Best weights updated")
                best_weights[k] = weights[k]
                metric_log["val_acc"][k] = val_acc
                metric_log["val_f1"][k] = val_f1
                metric_log["val_precision"][k] = val_p
                metric_log["val_recall"][k] = val_r
                metric_log["val_loss"][k] = output_dict["val_loss"]

    if save_fig:
        exp_name = f'{model.upper()}+degree-{degree}'

        for param, v in cv_params.items():
            exp_name += f'+{param}-{v}'
        exp_log_dir = os.path.join('log', exp_name)
        if not os.path.exists(exp_log_dir):
            os.mkdir(exp_log_dir)

        plot_training_stats(kfold_training_tracker=training_tracker,
                            save_path=f'{exp_log_dir}/training_stats.png',
                            title=exp_name)

        feat_weights_dict = {f: w for f, w in zip(
            feature_names, np.mean(best_weights, axis=0))}
        plot_weights(feat_weights_dict,
                     save_path=f'{exp_log_dir}/feature_weights.png')

    for key in metric_log.keys():
        metric_log[key] = np.mean(metric_log[key])

    return weights, best_weights, metric_log


class Trainer:
    """Trainer class for training and evaluating models"""
    def __init__(
        self,
        tx,
        y,
        params=None,
        init_w=None,
        k_fold=10,
        model="reg_logistic",
        save_fig=False,
        degree=4,
        model_name="reg_logistic",
    ):
        self.y = y
        self.tx = tx
        self.params = params
        self.k_fold = k_fold
        self.model = model
        self.degree = degree
        self.save_fig = save_fig
        self.monitor = params['monitor'][0]
        self.weights = [init_w for i in range(k_fold)]
        self.param_grid = build_parameter_grid(params)
        self.model_name = model_name
        self.checkpoint = f"./log/{model_name}_{k_fold}fold_cv_best.json"

    def train(self, lr_decay=False, feature_names=None):
        """
        Train the model
        
        :param lr_decay: whether to decay learning rate
        :param feature_names: feature names
        """
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
                save_fig=self.save_fig,
                degree=self.degree,
                feature_names=feature_names,
                lr_decay=lr_decay
            )

            print(f"Finish Validation: {metric_log}")

            if metric_log[self.monitor] > best_metric:
                print(
                    f"New best {self.monitor} found: {metric_log[self.monitor]}")
                best_metric = metric_log[self.monitor]
                best_params = config

                if not os.path.exists('./log'):
                    os.mkdir('./log')
                write_json({
                    "best_params": best_params,
                    self.monitor: metric_log[self.monitor],
                    "k_best_weights": [w.tolist() for w in best_weights]
                }, self.checkpoint)

            print(f"current best params: {best_params}")

    def eval(self, tx_test, ids_test):
        """
        Evaluate the model on test set
        
        :param tx_test: test set
        :param ids_test: test set ids
        """
        print(f"Load best weights from {self.checkpoint}")
        checkpoint = read_json(self.checkpoint)
        best_weights = checkpoint["k_best_weights"]

        print(f"Test set evaluation")

        for k in range(self.k_fold):
            test_preds = predict_test(
                tx=tx_test,
                w=np.array(best_weights[k]),
                logistic=("logistic" in self.model)
            )
                    
            if not os.path.exists('./output'):
                os.mkdir('./output')

            create_csv_submission(ids_test, test_preds,
                                  f"./output/{self.model_name}_fold_{k}_test_out.csv")

        test_ids, test_preds_vote = horizontal_voting(
            fold=self.k_fold, model=self.model
        )
        

        create_csv_submission(test_ids, test_preds_vote,
                              f"./output/{self.model_name}_vote_test_out.csv")

        print(f"Finish test set evaluation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory of the train and test data")
    parser.add_argument("--model", type=str, default="reg_logistic",
                        help="machine learning model type")
    parser.add_argument("--k_fold", type=int, default=5,
                        help="number of folds for cross validation")
    parser.add_argument("--save_fig", default=False, action="store_true",
                        help="save figure")
    parser.add_argument("--lr_decay", default=False, action="store_true",
                        help="decay learning rate gradually")
    parser.add_argument("--poly_feature", default=False, action="store_true",
                        help="data uagmentation with polynomial features")
    parser.add_argument("--do_train", default=False, action="store_true")
    parser.add_argument("--do_eval", default=False, action="store_true")
    parser.add_argument("--imputation", type=str, default="median",
                        help="Method for missing value imputation")
    parser.add_argument("--degree", type=int, default=4,
                        help="degree of polynomial feature augmentation")
    args = parser.parse_args()

    print("loading data ...")
    train_dataset = Dataset(
        data_pth=args.data_dir,
        data_type="train",
        poly_degree=args.degree,
        imputation=args.imputation
    )

    train_dataset.load_data(
        poly=args.poly_feature)

    random_seed = 42
    np.random.seed(random_seed)

    ids = train_dataset.ids
    labels = train_dataset.labels
    features = train_dataset.full_data
    shuffle_idx = np.random.permutation(np.arange(len(labels)))
    shuffled_y = labels[shuffle_idx]
    shuffled_tx = features[shuffle_idx]

    feature_names = train_dataset.read_col_names()

    init_w = np.random.uniform(low=-2.0, high=2.0, size=features.shape[1])
    model = args.model
    gd_gamma = [0.01, 0.05, 0.1, 0.25, 0.5]
    sgd_gamma = [0.01, 1e-3, 0.05, 1e-4, 5e-3]
    k_fold = args.k_fold
    batch_size = 100

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

    model_name = args.model
    if args.poly_feature:
        model_name += "_poly"
    if args.lr_decay:
        model_name += "_lr-decay"
        hyper_params["lr_schedule"] = ["linear", "epoch"]
        hyper_params["lambda"] = [1e-6]
        hyper_params["epochs"] = [20]
        hyper_params["gamma"] = [0.05]

    trainer = Trainer(
        tx=shuffled_tx,
        y=shuffled_y,
        params=hyper_params,
        init_w=init_w,
        k_fold=k_fold,
        model=model,
        save_fig=args.save_fig,
        degree=args.degree,
        model_name=model_name,
    )

    if args.do_train:
        trainer.train(
            lr_decay=args.lr_decay,
            feature_names=feature_names
        )

    if args.do_eval:
        test_dataset = Dataset(
            data_pth=args.data_dir,
            data_type="test",
            poly_degree=args.degree,
            imputation=args.imputation
        )
        test_dataset.load_data(
            poly=args.poly_feature)

        test_features = test_dataset.full_data
        test_ids = test_dataset.ids

        trainer.eval(test_features, test_ids)
