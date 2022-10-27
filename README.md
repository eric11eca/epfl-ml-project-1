# EPFL ML Course, Fall 2022, Project 1

Repository for the project 1 - data, code temlpates, and the report.

## Getting started 

We tested the code on python 3.9. 

### Requirements

  `numpy`

### Overview of Codebase

You can train the models by run the code `run.py` by 

  `python run.py --data_dir="./data/train.csv" --model="mse_gd" --k_fold=5 --do_train --poly_degree=4`
  
Then, evaluate the test data with the best weight of the model and make the submission file by `run.py` by
  
   `python run.py --data_dir="./data/test.csv" --model="mse_gd" --k_fold=5 --do_eval --poly_degree=4`

#### ML Implementations

You can find all the ML methods in `implementations.py`.
 
| ML | model args          | Arguments |
|-----------|--------------------|-----------|
|`mean_squared_error_gd`| `mse_gd`  | `y, tx, initial_w, max_iters, gamma`  | 
|`mean_squared_error_sgd`| `mse_sgd` | `y, tx, initial_w, max_iters, gamma, batch_size=1`  |
|`least_squares`| `least_squares`     | `y, tx` |
|`ridge_regression`| `ridge`  | `y, tx, lambda_` |
|`logistic_regression`| `logistic`| `y, tx, initial_w, max_iters, gamma, batch_size=1, sgd=False` |
|`reg_logistic_regression`| `reg_logistic` | `y, tx, lambda_, initial_w, max_iters, gamma, batch_size=1, sgd=False` |



