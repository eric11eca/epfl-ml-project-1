# EPFL CS433, Project 1

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

For all experiments, we save the best weight of the models with json as `./log/{model}_{k_fold}fold_cv_best.json`. And we used majority voting ensemble to finalize the prediction across the k-fold validation predictions, which is saved as `./output/{model}_test_majority.csv` for submission. 


#### Data preprocessing

You can find all the data preprocessing, feature generation, and feature selection steps in `scripts/dataset.py`. Details are as follows.

a) Data preparation:

We conducted one-hot encoding, data imputation, normalization, and outlier filtering.

- category_feature(): converted categorical feature into one-hot encoding

- data_imputation(): replaced missing values into mean/median, label mean/median

- data_normalization(): normalized each data point by - mean, and / by std

- filter_outliers(): filtered out outliers over mean +/- m * std

b) Feature generation:

We conducted feature augmentation by expanding each feature value into polynomial series.

- data_polynomial(): augmented feature by varying degrees


#### ML Implementations

You can find all the ML methods in `implementations.py`.
 
| ML | model args          | Loss | Parameters |
|-----------|--------------------|-----------|-----------|
| mean_squared_error_gd | `mse_gd`  | mse | `y, tx, initial_w, max_iters, gamma`  | 
| mean_squared_error_sgd | `mse_sgd` | mse | `y, tx, initial_w, max_iters, gamma, batch_size=1`  |
| least_squares | `least_squares`     | rmse | `y, tx` |
| ridge_regression | `ridge`  | rmse | `y, tx, lambda_` |
| logistic_regression | `logistic`| logistic | `y, tx, initial_w, max_iters, gamma, batch_size=1, sgd=False` |
| reg_logistic_regression | `reg_logistic` | logistic | `y, tx, lambda_, initial_w, max_iters, gamma, batch_size=1, sgd=False` |

For the performance improvement, we chose the best performance model (i.e., regularized logistic regression) with dynamic learning rate.

| ML | model args          | Loss | Parameters |
|-----------|--------------------|-----------|-----------|
| reg_logistic_dynamic | `reg_logistic_dynamic` | logistic | `y, tx, y_valid, tx_valid, initial_w, max_epoch_iters, gamma, batch_size=1, lambda_, dynamic_lr=True, k_cross=10, half_lr_count=2, early_stop_count=4` |



#### Cross-validation
    
We tested the models on 5-fold cross validation for finding the best hyperparameters (i.e., cv_params={gamma, lambda_}), which returns k-final weights, k-best weights, and logs includes training loss, validation logg, train accuracy, validation accuracy, validation f1 score, validation precision, and validation recall.

```
cross_validation(tx, y, initial_w, k_fold, cv_params, model, save_fig, degree, feature_names, lr_decay)
```

