import numpy as np
import pandas as pd

from scipy import stats

from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

from catboost import CatBoostRegressor, Pool

import lightgbm as lgb
from lightgbm import Dataset, LGBMRegressor

import xgboost as xgb
from xgboost import XGBRegressor

from consts import RANDOM_STATE, ModelType
from data import prepare_data

import warnings

warnings.filterwarnings('ignore')


def train_model(
        algorithm, X, y, early_stopping_rounds, init_params=None, cat_features=None, random_seed=2025, verbose=False,
        # threads_num=1,
) -> tuple[float, ModelType]:
    scores = []
    models = []

    kf = KFold(n_splits=3, shuffle=True, random_state=random_seed)

    print(f"========= TRAINING {algorithm.__name__} =========")

    for num_fold, (train_index, val_index) in enumerate(kf.split(X)):
        X_train, X_eval = X.iloc[train_index], X.iloc[val_index]
        y_train, y_eval = y.iloc[train_index], y.iloc[val_index]

        if init_params is not None:
            model = algorithm(**init_params)
        else:
            model = algorithm()

        if algorithm.__name__ == 'CatBoostRegressor':
            train_dataset = Pool(data=X_train, label=y_train, cat_features=cat_features)
            eval_dataset = Pool(data=X_eval, label=y_eval, cat_features=cat_features)

            model.fit(
                train_dataset,
                eval_set=eval_dataset,
                verbose=verbose,
                early_stopping_rounds=early_stopping_rounds,
                use_best_model=True
            )

        elif algorithm.__name__ == 'LGBMRegressor':
            # transform categorical features
            train_dataset = Dataset(X_train, y_train, categorical_feature=cat_features, free_raw_data=False)
            eval_dataset = Dataset(X_eval, y_eval, categorical_feature=cat_features, free_raw_data=False)

            init_params.update({'early_stopping_round': early_stopping_rounds})

            model = lgb.train(
                params=init_params,
                train_set=train_dataset,
                valid_sets=[eval_dataset, ],
            )

        elif algorithm.__name__ == 'XGBRegressor':
            # X_train[cat_features] = X_train[cat_features].astype('category')
            # X_eval[cat_features] = X_eval[cat_features].astype('category')

            train_dataset = xgb.DMatrix(
                X_train, label=y_train, enable_categorical=True # TODO nthread=threads_num,
            )
            eval_dataset = xgb.DMatrix(X_eval, label=y_eval)

            model = xgb.train(
                params=init_params,
                dtrain=train_dataset,
                evals=[(train_dataset, 'dtrain'), (eval_dataset, 'dtest')],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval= 10 if verbose else False
            )

            X_eval = eval_dataset

        if algorithm.__name__ == 'XGBRegressor':
            y_pred = model.predict(X_eval, iteration_range=(0, model.best_iteration + 1))
        else:
            y_pred = model.predict(X_eval)  # for xgboost iteration_range=(0, booster.best_iteration + 1)
        score = mean_squared_error(y_eval, y_pred) ** 0.5  # RMSE

        models.append(model)
        scores.append(score)

        print(f'FOLD {num_fold}: SCORE {score:.2f}')

    mean_kfold_score = np.mean(scores, dtype="float16")
    std_score = np.std(scores, dtype="float16")
    print(f"Mean RMSE score: {mean_kfold_score:.2f} +-{std_score:.2f} \n")
    best_model = models[np.argmin(scores)]

    return mean_kfold_score, best_model


def train_catboost(train, test) -> dict:
    X, y, X_test, _, _, cat_features = prepare_data(train, test)

    cb_init_params = {
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'thread_count': -1,
        'task_type': 'CPU',
        'random_seed': RANDOM_STATE
    }

    cb_score, cb_model = train_model(
        algorithm=CatBoostRegressor,
        X=X, y=y,
        init_params=cb_init_params,
        early_stopping_rounds=5,
        cat_features=cat_features,
        random_seed=RANDOM_STATE
    )

    cb_test_pred = cb_model.predict(X_test)
    pd.DataFrame({'car_id': test['car_id'], 'target_reg': cb_test_pred}).to_csv('cb_pred.csv', index=False)

    return {
        'model_name': 'CatBoostRegressor',
        'tuning': False,
        'kfold_score': cb_score,
        # 'leaderboard_score': ...,
        'model': cb_model
    }


def train_lightgbm(train, test) -> dict:
    X, y, X_test, cat_features = prepare_data(train, test)

    lgb_init_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'seed': RANDOM_STATE
    }

    lgb_score, lgb_model = train_model(
        algorithm=LGBMRegressor,
        X=X, y=y,
        init_params=lgb_init_params,
        early_stopping_rounds=5,
        cat_features=cat_features,
        random_seed=RANDOM_STATE
    )

    lgb_test_pred = lgb_model.predict(X_test)
    pd.DataFrame({'car_id': test['car_id'], 'target_reg': lgb_test_pred}).to_csv('lgb_pred.csv', index=False)

    return {
        'model_name': 'LGBMRegressor',
        'tuning': False,
        'kfold_score': lgb_score,
        # 'leaderboard_score': ...,
        'model': lgb_model
    }


def train_xgboost(train, test) -> dict:
    X, y, X_test, cat_features = prepare_data(train, test)

    xgb_init_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'verbosity': 0,
        'seed': RANDOM_STATE
    }

    xgb_score, xgb_model = train_model(
        algorithm=XGBRegressor,
        X=X, y=y,
        init_params=xgb_init_params,
        early_stopping_rounds=5,
        cat_features=cat_features,
        random_seed=RANDOM_STATE
    )

    xgb_test_pred = xgb_model.predict(X_test)
    pd.DataFrame({'car_id': test['car_id'], 'target_reg': xgb_test_pred}).to_csv('xgb_pred.csv', index=False)

    return {
        'model_name': 'XGBRegressor',
        'tuning': False,
        'kfold_score': xgb_score,
        # 'leaderboard_score': ...,
        'model': xgb_model
    }
