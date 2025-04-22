from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score #, mean_squared_error
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import seaborn as sns
import numpy as np

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def load_data():
    path = './data/car_train.csv'
    train = pd.read_csv(path)

    path = './data/car_test.csv'
    test = pd.read_csv(path)

    features2drop = ['car_id']
    targets = ['target_class', 'target_reg']
    cat_features = ['car_type', 'fuel_type', 'model']

    filtered_features = [i for i in train.columns if (i not in targets and i not in features2drop)]
    num_features = [i for i in filtered_features if i not in cat_features]

    print('cat_features :', len(cat_features), cat_features)
    print('num_features :', len(num_features), num_features)
    print('targets', targets)

    X = train[filtered_features]
    y = train['target_class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, cat_features, test[filtered_features]


def get_baseline(X_train, X_test, y_train, y_test, cat_features):
    model = CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=6, loss_function='MultiClass')
    model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_test, y_test), verbose=100)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print('Accuracy:', acc)


