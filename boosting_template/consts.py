from catboost import CatBoostRegressor, Pool
from lightgbm import Dataset, Booster, LGBMRegressor
from xgboost import XGBRegressor

from typing import Union
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="darkgrid")

RANDOM_STATE = 42

BEST_FOLD_RMSE = float
MEAN_RMSE = float
STD_RMSE = float
ModelType = Union[CatBoostRegressor, Booster, XGBRegressor]


catboost_grid_params = {
    'depth': [4, 6, 8],  # tree depth
    'l2_leaf_reg': [1, 3, 5],  # L2 regularization coefficient
    'bagging_temperature': [0.5, 1.0, 1.5],  # bagging temperature
    'random_strength': [0.1, 0.5, 1.0],  # randomness when selecting features
    'one_hot_max_size': [2, 5, 10],  # threshold for one-hot encoding
    'colsample_bylevel': [0.5, 0.8, 1.0],  # fraction of columns for each tree level
    'leaf_estimation_iterations': [5, 10, 15],  # number of iterations to optimize leaf values
    'max_ctr_complexity': [1, 3, 5],  # complexity of categorical features
    'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide'],  # tree growth policy
    'min_data_in_leaf': [3, 5, 7, 11],  # minimum number of samples in a leaf
    'leaf_estimation_method': ['Newton', 'Gradient'],  # method for leaf value estimation
    'max_leaves': [30, 50, 100],  # maximum number of leaves (for Lossguide policy)
    'max_bin': [128, 256, 512],  # number of bins for numerical features
    'boosting_type': ['Plain', 'Ordered'],  # type of boosting
    'bootstrap_type': ['No', 'Bayesian', 'Bernoulli', 'Poisson', 'MVS'],  # bootstrap method
    'random_state': [42],  # random seed for reproducibility
}


lgb_grid_params = {
    'num_leaves': [31, 50, 100],  # controls the complexity of the model.
    'max_depth': [-1, 10, 20, 30],  # maximum depth of a tree; -1 means no limit.
    'learning_rate': [0.01, 0.05, 0.1, 0.2],  # shrinks the contribution of each tree.
    'n_estimators': [100, 200, 300],  # number of boosting iterations.
    'min_child_samples': [10, 20, 50],  # minimum data in a leaf node.
    'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1],  # minimum sum of instance weight in a leaf.
    'subsample': [0.6, 0.8, 1.0],  # fraction of samples used for training each tree.
    'colsample_bytree': [0.6, 0.8, 1.0],  # fraction of features used for training each tree.
    'reg_alpha': [0, 0.01, 0.1],  # L1 regularization term on weights.
    'reg_lambda': [0, 0.01, 0.1],  # L2 regularization term on weights.
}


xgb_grid_params = {
    'objective': ['reg:squarederror'],  # objective function for optimization
    'eval_metric': ['rmse'],  # metric to evaluate model quality
    'eta': [0.05, 0.1, 0.2],  # learning rate of the model
    'booster': ['dart'],  # boosting method
    'rate_drop': [0.1, 0.05, 0.5],  # dropout probability for each tree during training
    'skip_drop': [0.5, 0.2, 0.8],  # probability to skip dropout for each tree during training
    'max_depth': [3, 5, 6, 7],  # maximum depth of a tree
    'subsample': [0.8, 0.9, 1.0],  # fraction of training instances used to build each tree
    'colsample_bytree': [0.8, 0.9, 1.0],  # fraction of features used to build each tree
    'colsample_bylevel': [0.8, 0.9, 1.0],  # fraction of features used at each tree level
    'colsample_bynode': [0.8, 0.9, 1.0],  # fraction of features used at each tree node split
    'lambda': [0.1, 1, 5, 10, 12],  # L2 regularization term on weights
    'alpha': [0.1],  # L1 regularization term on weights
    'gamma': [0, 0.1],  # minimum loss reduction required to make a further partition on a leaf node
    'tree_method': ['hist'],  # tree construction algorithm
    'min_child_weight': [1, 5, 10],  # minimum sum of instance weight needed in a child
    'max_bin': [256, 512, 1024],  # maximum number of bins for discretizing continuous features
    'seed': [42]  # random seed for reproducibility
}

