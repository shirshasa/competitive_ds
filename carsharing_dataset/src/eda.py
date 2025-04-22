import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data(file_path):
    return pd.read_csv(file_path)


def check_basic_info(df):
    target = 'target_reg'
    category_columns = df.select_dtypes(include=['object']).columns
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

    print("Shape of the data is: ", df.shape)
    print("Columns in the data are: ", df.columns)
    print("Data types of the columns are: ", df.dtypes)
    print("Basic statistics of the data are: ", df.describe())

    print("Checking for missing values: ", df.isnull().sum())
    print("Checking for duplicate values: ", df.duplicated().sum(axis=0))
    print("Checking for unique values: ", df.nunique())
    print("Checking zeros count: ", (df[numerical_columns] == 0).sum(axis=0))
    print("Checking cor: ", df[numerical_columns].corr())

    df.hist(figsize=(20, 5), layout=(-1, 5))
    plt.show()

    # plot values counts on one subplot
    fig, ax = plt.subplots(1, len(category_columns), figsize=(20, 5))
    for i, feat in enumerate(category_columns):
        df[feat].value_counts().plot(kind='bar', ax=ax[i])
        ax[i].set_title(feat)

    plt.show()

    # check for outliers
    for feat in numerical_columns:
        df.boxplot(column=feat)
        plt.title(feat)
        plt.show()

    # check for distribution
    for feat in numerical_columns:
        df[feat].plot(kind='density')
        plt.title(feat)
        plt.show()

    # check cor with target
    for feat in numerical_columns:
        df.plot.scatter(x=feat, y=target)
        plt.title(feat)
        plt.show()

    for feat in category_columns:
        df.groupby(feat)[target].mean().plot(kind='bar')
        plt.title(feat)
        plt.show()


if __name__ == "__main__":
    file_path = "/data/projects/4_kaggle/carsharing/data/car_train.csv"
    df = load_data(file_path)
    check_basic_info(df)

