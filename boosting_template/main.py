import numpy as np
import pandas as pd
import xgboost as xgb

import warnings

warnings.filterwarnings('ignore')

from consts import *
from data import load_data, prepare_data
from hp_tunning import train_lightgbm_hp, train_xgboost_hp


train, test = load_data()
X, y, X_test, X_encoded, X_test_encoded, cat_features = prepare_data(train, test)

xgb_init_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'verbosity': 0,  # 0 (silent), 1 (warning), 2 (info), 3 (debug)
    'seed': RANDOM_STATE,
    'enable_categorical': True,
    'nthread': -1
}


grid_params = {
    'num_boost_round': [50],
    'max_depth': [3, 5, 6,],
    'max_leaves': [7, 9, 11, 15],
    'min_child_weight': [12, 14, 16, 18],
    'lambda': [0.05, 0.1, 1, 5, 10, 12],
    'rate_drop': [0.05, 0.1, 0.25, 0.3],
    'one_drop': [0, 1],
    'skip_drop': [0.1, 0.25, 0.3, 0.5],
    'booster': ['dart'],
    'eta': [0.1, 0.15, 0.2, 0.25,],
}

# learning_rates = np.linspace(0.3, 0.005, 50).tolist()
# scheduler = xgb.callback.LearningRateScheduler(learning_rates)
# fit_params = dict(callbacks=[scheduler])

tuning_score, tuning_model = train_xgboost_hp(
        X_encoded, y, xgb_init_params, grid_params, {}, cat_features,
        n_iter=5,
        cv=3,
        verbose=False,
        random_seed=RANDOM_STATE,
        n_jobs=1
)

X_test_dmatrix = xgb.DMatrix(X_test_encoded, enable_categorical=True)
tuning_test_pred = tuning_model.predict(X_test_dmatrix, iteration_range=(0, tuning_model.best_iteration + 1))
pd.DataFrame({'car_id': test['car_id'], 'target_reg': tuning_test_pred}).to_csv('subs/xgb_pred_hp.csv', index=False)
