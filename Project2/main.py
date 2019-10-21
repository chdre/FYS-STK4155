import numpy as np
import sklearn.linear_model as skl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import seaborn as sns
import tensorflow as tf
import pandas as pd
import os
from tabulate import tabulate


np.random.seed(0)


def import_data():
    np.random.seed(0)

    cwd = os.getcwd()   # Current working directory
    filename = cwd + '/data/default of credit card clients.xls'
    nanDict = {}    # To store NaN from CC data when reading with pandas?
    df = pd.read_excel(filename, header=1, index_col=0,
                       skiprows=0, na_values=nanDict)  # Dataframe

    # Renaming axis
    df.rename(index=str,
              columns={df.columns[-1]: 'DefaultPaymentNextMonth'},
              inplace=True)

    # Features and targets
    x = df.loc[:, df.columns != 'DefaultPaymentNextMonth'].values
    y = df.loc[:, df.columns == 'DefaultPaymentNextMonth'].values

    # Design matrix
    X = np.c_[np.array([1] * len(x[:, 0])), X]

    # Train-test split
    train_size_ = 0.8
    xtest, xtrain, ytest, ytrain = train_test_split(
        x, y, train_size=train_size_, test_size=1 - train_size_)

    # Scaling
    scale = StandardScaler()   # Scales by (func - mean)/std.dev
    xtrain = scale.fit_transform(xtrain)
    xtest = scale.transform(xtest)

    df = df.drop(df[(df.BILL_AMT1 == 0) &
                    (df.BILL_AMT2 == 0) &
                    (df.BILL_AMT3 == 0) &
                    (df.BILL_AMT4 == 0) &
                    (df.BILL_AMT5 == 0) &
                    (df.BILL_AMT6 == 0)].index)

    df = df.drop(df[(df.PAY_AMT1 == 0) &
                    (df.PAY_AMT2 == 0) &
                    (df.PAY_AMT3 == 0) &
                    (df.PAY_AMT4 == 0) &
                    (df.PAY_AMT5 == 0) &
                    (df.PAY_AMT6 == 0)].index)

    return xtrain, xtest,


def cost_function(X, y, beta):
    """costfunc"""
    return np.sum(y @ (X @ beta) - np.log(1 + np.exp(X @ beta)))


def stochastic_gradient_descent(X, y, beta,
                                eps=1e-5, n=100, eta=0.1,
                                mini_batch=False, n_epochs=50):  # stochastic GD
    """ stochastic gradient descent"""
    if mini_batch:
        for e in range(n_epochs):
            for i in range(n):
                random_idx = np.random.randint(n)
                xi = X[random_idx:random_idx + 1]
    for i in range(n):
        beta_new = beta - eta *
        if abs(np.sum(beta - beta_new)) < eps:
            return beta_new
        beta = beta_new


def main():

    beta = np.random.randn()


if __name__ == '__main__':
    import_data()
