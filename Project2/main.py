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

    def learning_schedule(t, t0=5, t1=50): return t0 / (t + t1)

    if mini_batch:
        for epoch in range(n_epochs):
            while np.abs(beta_old - beta_new) < eps or iter < n:
                rand_idx1 = np.random.randint(int(n / 2))
                rand_idx2 = np.random.randint(int(n / 2))
                xi = X[rand_idx1:rand_idx + 1,
                       rand_idx2:rand_idx2 + 1]  # Matrix
                yi = y[rand_idx1:rand_idx1 + 1]  # Array
                P = np.exp(X @ beta) / (1 + np.exp(X @ beta))
                gradient = X.T @ (P - y)
                eta = learning_schedule(epoch * m + i)
                beta -= eta * gradient

            beta_old = beta
            beta = beta_new

        return beta_new

    if not mini_batch:
        for i in range(n):
            P = np.exp(X @ beta) / (1 + np.exp(X @ beta))
            gradient = X.T  @ (P - y)
            beta_new = beta - eta * gradient

            if abs(np.sum(beta - beta_new)) < eps:
                return beta_new

            beta_old = beta
            beta = beta_new

    return beta


def main():

    beta = np.random.randn(len(x[0]), 1)


if __name__ == '__main__':
    import_data()
