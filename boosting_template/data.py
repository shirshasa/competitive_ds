import os.path

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder


import warnings

warnings.filterwarnings('ignore')


def load_data():
    if os.path.exists('data/quickstart_train.csv'):
        train = pd.read_csv('data/quickstart_train.csv', index_col=0)
    else:
        train = pd.read_csv(
        'https://raw.githubusercontent.com/a-milenkin/Competitive_Data_Science/main/data/quickstart_train.csv')

    if os.path.exists('data/quickstart_test.csv'):
        test = pd.read_csv('data/quickstart_test.csv', index_col=0)
    else:
        test = pd.read_csv(
            'https://raw.githubusercontent.com/a-milenkin/Competitive_Data_Science/main/data/quickstart_test.csv')

    return train, test


def prepare_data(train, test):
    cat_features = ['user_uniq', 'model', 'car_type', 'fuel_type']
    targets = ['target_reg', 'target_class']
    features2drop = ['car_id', 'deviation_normal_count']

    filtered_features = [f for f in train.columns if f not in targets + features2drop]
    num_features = [f for f in filtered_features if f not in cat_features]

    print("cat_features", cat_features)
    print("num_features", num_features)
    print("targets", targets, "\n")

    X = train[filtered_features].drop(targets, axis=1, errors="ignore")
    y = train["target_reg"]
    X_test = test[filtered_features]

    X_encoded = X.copy()
    X_test_encoded = X_test.copy()

    def transform_category_features(X, X_test):
        le = LabelEncoder()
        for col in cat_features:
            unique = X[col].unique()
            mode = X[col].mode()[0]
            X_test[col] = X_test[col].apply(lambda x: x if x in unique else mode)

            X[col] = le.fit_transform(X[col])
            X_test[col] = le.transform(X_test[col])

    transform_category_features(X_encoded, X_test_encoded)

    for col in cat_features:
        X_encoded[col] = X_encoded[col].astype('category')
        X_test_encoded[col] = X_test_encoded[col].astype('category')
    return X, y, X_test, X_encoded, X_test_encoded, cat_features
