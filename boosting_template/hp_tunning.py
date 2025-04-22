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

from consts import RANDOM_STATE, ModelType, BEST_FOLD_RMSE, MEAN_RMSE, STD_RMSE

import warnings

warnings.filterwarnings('ignore')


def tuning_hyperparams(
        algorithm, X, y, init_params, fit_params, grid_params, n_iter, cv=3, random_state=2025,
        verbose=False
):
    estimator = algorithm(**init_params)

    model = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=grid_params,
        n_iter=n_iter,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=verbose,
        random_state=random_state
    )
    model.fit(X, y, **fit_params)

    return model.best_params_ | init_params


def train_catboost_hp(
        X, y, init_params, params, cat_features, n_iter=10, cv=3, verbose=False, random_seed=RANDOM_STATE, plot=False
) -> tuple[BEST_FOLD_RMSE, MEAN_RMSE, STD_RMSE, ModelType]:
    assert 'early_stopping_rounds' in init_params

    kf = KFold(n_splits=3, shuffle=True, random_state=random_seed)

    models = []
    rmse_scores = []
    best_params = []

    for i, (train_index, val_index) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        val_pool = Pool(X_val, y_val, cat_features=cat_features)

        model = CatBoostRegressor(**init_params)
        search_results = model.randomized_search(
            params,
            X=train_pool,
            cv=cv,
            n_iter=n_iter,
            verbose=verbose,
            calc_cv_statistics=False,
            search_by_train_test_split=False,
            refit=True,
            shuffle=True,
            stratified=None,
            train_size=0.8,
            plot=plot,
            partition_random_seed=random_seed,
        )
        models.append(model)

        pred = model.predict(val_pool)
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        rmse_scores.append(rmse)

        pred_train = model.predict(train_pool)
        rmse_train = np.sqrt(mean_squared_error(y_train, pred_train))
        best_params.append(search_results['params'])

        print(f"BEST PARAMS for split {i}:\n", search_results['params'])
        print(f"SCORE val/train: {rmse:.2f}/{rmse_train:.2f} \n")

    best_model_index = np.argmin(rmse_scores)
    best_model = models[best_model_index]

    mean_kfold_score = np.mean(rmse_scores, dtype="float16")
    std_score = np.std(rmse_scores, dtype="float16")
    print(f"MEAN RMSE score: {mean_kfold_score:.2f} +-{std_score:.2f} \n")
    print("BEST MODEL:", best_params[best_model_index])
    print(f'BEST RMSE: {rmse_scores[best_model_index]:.2f} \n')

    return rmse_scores[best_model_index], mean_kfold_score, std_score, best_model


def train_lightgbm_hp(
        X, y, init_params, grid_params, fit_params, cat_features,
        n_iter=10,
        cv=3,
        verbose=False,
        random_seed=RANDOM_STATE,
        n_jobs=1
) -> tuple[BEST_FOLD_RMSE, MEAN_RMSE, STD_RMSE, ModelType]:
    """
    Example:
        >>> lgb_init_params = {
        ...     'objective': 'regression',
        ...     'metric': 'rmse',
        ...     'data_sample_strategy': 'goss',
        ...     'verbosity': -1,  # -1 for suppression
        ...     'seed': RANDOM_STATE,
        ...     'num_iterations': 50,
        ... }
        >>> lgb_fit_params = {
        ...     'eval_metric': 'rmse',
        ...     'categorical_feature': cat_features
        ... }
        >>> lgb_grid_params = {
        ...     'reg_alpha': [0.5, 0.75, 1],
        ...     'num_leaves': [4, 5, 6, 7],
        ...     'max_bin': [40, 50, ],
        ...     'num_iterations': [100],
        ...     'min_child_samples': [10, 11, 12, 15],
        ... }
        >>> lgb_tuning_score, lgb_tuning_model = train_lightgbm_hp(
        ...     X, y, lgb_init_params, lgb_grid_params, lgb_fit_params, cat_features,
        ...     n_iter=50,
        ...     cv=3,
        ...     verbose=0,
        ...     random_seed=RANDOM_STATE
        ... )
        >>> lgb_test_pred = lgb_tuning_model.predict(X_test)
        >>> pd.DataFrame({'car_id': test['car_id'], 'target_reg': lgb_test_pred}).to_csv('subs/lgb_pred_hp.csv', index=False)

    """

    kf = KFold(n_splits=3, shuffle=True, random_state=random_seed)

    models = []
    rmse_scores = []
    best_params = []

    for i, (train_index, val_index) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features, free_raw_data=False)
        valid_data = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_features, free_raw_data=False)

        model = lgb.LGBMRegressor(**init_params)

        random_search = RandomizedSearchCV(
            model,
            grid_params,
            n_iter=n_iter,
            cv=cv,
            scoring='neg_root_mean_squared_error',
            random_state=random_seed,
            verbose=verbose,
            n_jobs=n_jobs
        )
        random_search.fit(
            X_train, y_train,
            # callbacks = [
            #     lgb.log_evaluation(period=10)
            # ],
            **fit_params
        )

        callbacks = [
            lgb.early_stopping(10),
            lgb.log_evaluation(period=10)
        ]
        init_params.update(random_search.best_params_)

        model = lgb.train(
            init_params,
            train_data,
            valid_sets=[valid_data],
            # callbacks=callbacks,
        )
        models.append(model)
        best_params.append(init_params)

        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        rmse_scores.append(rmse)

        y_pred_train = model.predict(X_train)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))

        print(f"BEST PARAMS for split {i}:", best_params[i])
        print(f"SCORE val/train: {rmse:.2f}/{rmse_train:.2f}, ")

    best_model_index = np.argmin(rmse_scores)
    best_model = models[best_model_index]

    mean_kfold_score = np.mean(rmse_scores, dtype="float16")
    std_score = np.std(rmse_scores, dtype="float16")
    print(f"MEAN RMSE score: {mean_kfold_score:.2f} +-{std_score:.2f} \n")
    print("BEST MODEL:", best_params[best_model_index])
    print(f'BEST RMSE: {rmse_scores[best_model_index]:.2f}')

    return rmse_scores[best_model_index], mean_kfold_score, std_score, best_model


def train_xgboost_hp(
        X, y, init_params, grid_params, fit_params, cat_features,
        n_iter=10,
        cv=3,
        verbose=False,
        random_seed=RANDOM_STATE,
        n_jobs=1
) -> tuple[BEST_FOLD_RMSE, MEAN_RMSE, STD_RMSE, ModelType]:
    """
    Example:
        >>> xgb_init_params = {
        ...     'objective': 'reg:squarederror',
        ...     'eval_metric': 'rmse',
        ...     'verbosity': 0,  # 0 (silent), 1 (warning), 2 (info), 3 (debug)
        ...     'seed': RANDOM_STATE,
        ...     'enable_categorical': True,
        ...     'nthread': -1
        ... }
        >>> grid_params = {
        ...     'num_boost_round': [100],
        ...     'max_depth': [3, 5],
        ...     'max_leaves': [7, 9, 10],
        ...     'min_child_weight': [10, 12, 14, 16],
        ...     'lambda': [0.05, 0.1, 1, 5, 10, 12],
        ...     'rate_drop': [0.05, 0.1, 0.25, 0.3],
        ...     'skip_drop': [0.1, 0.25, 0.3, 0.5],
        ...     'booster': ['dart'],
        ...     'eta': [0.05, 0.1, 0.2,],
        ... }
        ...
        >>> learning_rates = np.linspace(0.3, 0.005, 50).tolist()
        >>> scheduler = xgb.callback.LearningRateScheduler(learning_rates)
        >>> fit_params = dict(callbacks=[scheduler])
        ...
        >>> tuning_score, tuning_model = train_xgboost_hp(
        ...     X_encoded, y, xgb_init_params, grid_params, fit_params, cat_features,
        ...     n_iter=40,
        ...     cv=3,
        ...     verbose=False,
        ...     random_seed=RANDOM_STATE,
        ...     n_jobs=1
        ... )
        >>> X_test_dmatrix = xgb.DMatrix(X_test_encoded, enable_categorical=True)
        >>> tuning_test_pred = tuning_model.predict(
        ...     X_test_dmatrix, iteration_range=(0, tuning_model.best_iteration + 1)
        ... )
        >>> pd.DataFrame({'car_id': test['car_id'], 'target_reg': tuning_test_pred}).to_csv('subs/xgb_pred_hp.csv', index=False)
    """

    assert 'num_boost_round' in grid_params

    kf = KFold(n_splits=3, shuffle=True, random_state=random_seed)
    models = []
    rmse_scores = []
    best_params = []

    for i, (train_index, val_index) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        train_data = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        valid_data = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)

        model = XGBRegressor(**init_params)
        random_search = RandomizedSearchCV(
            model, grid_params, n_iter=n_iter, cv=cv,
            scoring='neg_root_mean_squared_error',
            random_state=random_seed,
            verbose=verbose,
            n_jobs=n_jobs
        )
        random_search.fit(X_train, y_train)

        init_params.update(random_search.best_params_)
        best_params.append(random_search.best_params_)
        print(f"TRAINING FINAL {i} FOLD MODEL..")
        model = xgb.train(
            init_params,
            train_data,
            num_boost_round=1000,
            evals=[(valid_data, "validation")],
            early_stopping_rounds=10,
            verbose_eval=10
        )

        models.append(model)
        y_pred = model.predict(valid_data, iteration_range=(0, model.best_iteration + 1))
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        rmse_scores.append(rmse)

        y_pred_train = model.predict(train_data, iteration_range=(0, model.best_iteration + 1))
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))

        print(f"BEST ITER: {model.best_iteration}")
        print(f"BEST PARAMS for split {i}:\n", random_search.best_params_)
        print(f"SCORE val/train: {rmse:.2f}/{rmse_train:.2f}\n")

    best_model_index = np.argmin(rmse_scores)
    best_model = models[best_model_index]

    mean_kfold_score = np.mean(rmse_scores, dtype="float16")
    std_score = np.std(rmse_scores, dtype="float16")
    print(f"MEAN RMSE score: {mean_kfold_score:.2f} +-{std_score:.2f}")
    print("BEST MODEL:\n", best_params[best_model_index])
    print(f'BEST RMSE: {rmse_scores[best_model_index]:.2f}')

    return rmse_scores[best_model_index], mean_kfold_score, std_score, best_model

