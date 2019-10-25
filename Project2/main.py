import numpy as np
import sklearn.linear_model as skl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
import seaborn as sns
import tensorflow as tf
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


np.random.seed(42)


def data_import():
    """
    Imports credit card data, returns test and train set for design matrix and y.
    """
    # Importing data
    cwd = os.getcwd()   # Current working directory
    filename = cwd + '/data/default of credit card clients.xls'
    nanDict = {}    # To store NaN from CC data when reading with pandas?
    df = pd.read_excel(filename, header=1, index_col=0,
                       skiprows=0, na_values=nanDict)  # Dataframe

    # Dropping rows where no billing and no payment
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

    # Renaming axis
    df.rename(index=str,
              columns={df.columns[-1]: 'DefaultPaymentNextMonth'},
              inplace=True)

    # Features and targets
    x = df.loc[:, df.columns != 'DefaultPaymentNextMonth'].values
    y = df.loc[:, df.columns == 'DefaultPaymentNextMonth'].values

    # Categorical variables to one-hot's
    onehotencoder = OneHotEncoder(categories="auto", sparse=False)

    x = ColumnTransformer(
        [("", onehotencoder, [1, 2, 3, 5, 6, 7, 8, 9, 10])],
        remainder="passthrough"
    ).fit_transform(x)

    y_onehot = onehotencoder.fit_transform(y)

    return x, y


def gradient_descent(X, y, beta,
                     eps=1e-15, n=10000, eta=1e-6,
                     stochastic=False, n_epochs=100, m=500):  # stochastic GD
    """gradient descent"""
    def learning_schedule(t, t0=5, t1=50): return t0 / (t + t1)

    beta_old = beta + 1  # Initial value for while loop

    if stochastic:
        for epoch in range(n_epochs):
            iter = 0
            while np.abs(np.sum(beta - beta_old)) < eps or iter < m:
                rand_idx = np.random.randint(int(len(X[0])))
                xi = X[rand_idx:rand_idx + 1]  # Matrix
                yi = y[rand_idx:rand_idx + 1]  # Array

                sigmoid = 1 / (1 + np.exp(-xi @ beta))
                gradient = xi.T @ (sigmoid - yi)

                eta = learning_schedule(epoch * m + iter)
                beta_new = beta - eta * gradient

                beta_old = beta
                beta = beta_new

                iter += 1

        return beta

    if not stochastic:
        for i in range(n):
            sigmoid = 1 / (1 + np.exp(-X @ beta))
            gradient = X.T @ (sigmoid - y)
            beta_new = beta - eta * gradient

            if abs(np.sum(beta - beta_new)) < eps:
                print(f"Converged for i={i}")
                return beta_new

            beta_old = beta
            beta = beta_new

        return beta


def accuracy(y_tilde, y):
    """ returns accruacy"""
    I = np.mean(y_tilde == y)
    return I


def cost_function(prob, y):
    """costfunc"""
    return -np.sum(y @ np.log(prob) - np.log(1 + np.exp(X @ beta)))


def main():
    """ main """
    x, y = data_import()

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

    # Scaling
    scale = StandardScaler()   # Scales by (func - mean)/std.dev
    xtrain = scale.fit_transform(xtrain)
    xtest = scale.transform(xtest)

    Xtrain = np.c_[np.array([1] * len(xtrain[:, 0])), xtrain]
    Xtest = np.c_[np.array([1] * len(xtest[:, 0])), xtest]

    beta_init = np.random.randn(len(Xtrain[0]), 1)

    beta_GD = gradient_descent(Xtrain, ytrain, beta_init, n=10000)
    prob_GD = 1 / (1 + (np.exp(-Xtest @ beta_GD)))
    pred_GD = (prob_GD >= 0.5).astype(int)

    beta_SGD = gradient_descent(
        Xtrain, ytrain, beta_init, m=300, stochastic=True)
    prob_SGD = 1 / (1 + (np.exp(-Xtest @ beta_SGD)))
    pred_SGD = (prob_SGD >= 0.5).astype(int)

    clf = LogisticRegression(solver='lbfgs')
    clf_fit = clf.fit(xtrain, np.ravel(ytrain))
    pred_skl = xtest @ clf_fit.coef_.T

    print(f"Accuracy score for own GD: {accuracy(pred_GD, ytest)}")
    print(f"Accuracy score for own SGD: {accuracy(pred_SGD, ytest)}")
    print(f"Accuracy score scikit-learn: {clf.score(xtest, ytest)}")

    plot_confusion_matrix(ytest, pred_GD)
    plot_confusion_matrix(ytest, pred_SGD)
    plot_confusion_matrix(ytest, pred_skl)


def plot_confusion_matrix(y, pred):
    conf_matrix = confusion_matrix(y, pred)
    sns.heatmap(pd.DataFrame(conf_matrix), annot=True, cmap="YlGnBu", fmt='g')
    plt.title('Confusion matrix (default = 1)')
    plt.ylabel('True value')
    plt.xlabel('Predicted value')
    plt.show()


if __name__ == '__main__':
    main()
