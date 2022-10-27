# EPFL ML Course, Fall 2022, Project 1

Repository for the project 1 - data, code temlpates, and the report.

## Getting started 

We tested the code on python 3.9. 

### Requirements

  ```
  numpy
  ```

### Overview of Codebase

You can train the models by run the code `run.py` by 

  ```
  python run.py --data_dir="./data/train.csv" --model="mse_gd" --k_fold=5 --do_train --poly_degree=4
  ```
  
Then, evaluate the test data with the best weight of the model and make the submission file by `run.py` by
  
   ```
   python run.py --data_dir="./data/test.csv" --model="mse_gd" --k_fold=5 --do_eval --poly_degree=4
   ```

For all experiments, we save the best weight of the models with json as `./log/{model}_{k_fold}fold_cv_best.json`. And we used majority voting ensemble to finalize the predictions across the k-fold validations, which is saved as `./output/{model}_test_majority.csv` for submission. 

#### ML Implementations

You can find all the ML methods in `implementations.py`.
 
| ML | model args          | Parameters |
|-----------|--------------------|-----------|
| `mean_squared_error_gd` | `mse_gd`  | `y, tx, initial_w, max_iters, gamma`  | 
| `mean_squared_error_sgd` | `mse_sgd` | `y, tx, initial_w, max_iters, gamma, batch_size=1`  |
| `least_squares` | `least_squares`     | `y, tx` |
| `ridge_regression` | `ridge`  | `y, tx, lambda_` |
| `logistic_regression` | `logistic`| `y, tx, initial_w, max_iters, gamma, batch_size=1, sgd=False` |
| `reg_logistic_regression` | `reg_logistic` | `y, tx, lambda_, initial_w, max_iters, gamma, batch_size=1, sgd=False` |

For the performance improvement, we chose the best performance model (i.e., regularized logistic regression) with dynamic learning rate.

| ML | model args          | Parameters |
|-----------|--------------------|-----------|
| `reg_logistic_dynamic` | `reg_logistic_dynamic` | `y, tx, y_valid, tx_valid, initial_w, max_epoch_iters, gamma, batch_size=1,
                         lambda_, dynamic_lr=True, k_cross=10, half_lr_count=2, early_stop_count=4` |

#### Data preprocessing

You can find all the data preprocessing, feature generation, and feature selection steps in `dataset.py`. Details are as follows.

a) Data preparation:

```

```

b) Feature generation:

```

```

c) Feature selection:

```

```

#### Cross-validation

