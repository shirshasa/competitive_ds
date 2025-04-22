import numpy as np
import pandas as pd

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
) -> tuple[float, float, float, ModelType]:
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
            train_dataset = xgb.DMatrix(
                X_train, label=y_train, enable_categorical=True  # TODO nthread=threads_num,
            )
            eval_dataset = xgb.DMatrix(X_eval, label=y_eval, enable_categorical=True)

            model = xgb.train(
                params=init_params,
                dtrain=train_dataset,
                evals=[(train_dataset, 'dtrain'), (eval_dataset, 'dtest')],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=10 if verbose else False
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

    return scores[np.argmin(scores)], mean_kfold_score, std_score, best_model

