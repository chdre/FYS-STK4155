import numpy as np
import seaborn as sns
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix


def plot_heatmap(matrix, xtick, ytick):
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(matrix, annot=True, ax=ax, cmap="viridis",
                xticklabels=xtick, yticklabels=ytick)
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.show()


def plot_confusion_matrix(y, pred):
    conf_matrix = confusion_matrix(y, pred)
    sns.heatmap(pd.DataFrame(conf_matrix), annot=True, cmap="YlGnBu", fmt='g')
    # plt.title('Confusion matrix (default = 1)')
    plt.ylabel('True value')
    plt.xlabel('Predicted value')
    plt.show()


def scale_data(x, y, method):
    scale = method()
    scale.fit(x)
    x = scale.transform(x)
    y = scale.transform(y)
    return x, y


def learning_schedule(t, t0=5, t1=50):
    return t0 / (t + t1)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def Franke_function(x, y):
    t1 = 0.75 * np.exp(-((9 * x - 2)**2) / 4 - ((9 * y - 2)**2) / 4)
    t2 = 0.75 * np.exp(-((9 * x + 1)**2) / 49 - ((9 * y + 1)**2) / 10)
    t3 = 0.5 * np.exp(-((9 * x - 7)**2) / 4 - ((9 * y - 3)**2) / 4)
    t4 = -0.2 * np.exp(-(9 * x - 4)**2 - (9 * y - 7)**2)

    f = t1 + t2 + t3 + t4

    return f


def credit_card_data_import(plot_corr=False):
    """
    Imports credit card data, returns test and train set for design matrix and y.
    X is onehot encoded.
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

    # Scaling x (second argument)
    x, _ = scale_data(x, x, StandardScaler)

    # Categorical variables to one-hot's
    onehotencoder = OneHotEncoder(categories="auto", sparse=False)

    x = ColumnTransformer(
        [("", onehotencoder, [1, 2, 3, 5, 6, 7, 8, 9, 10])],
        remainder="passthrough"
    ).fit_transform(x)

    y_onehot = onehotencoder.fit_transform(y)

    if plot_corr:
        df_scaled = df - df.mean()
        corr = df_scaled.corr()
        sns.heatmap(corr, annot=True, fmt='.2f')
        plt.xticks(rotation=90)
        # Because turning something 360 degrees helps??? :)
        plt.yticks(rotation=360)
        plt.show()

    return x, y, y_onehot


def gradient_descent(X, y, beta, eps=1e-6, n=10000, eta=1e-6):
    """gradient descent"""
    beta_old = beta

    for i in range(n):
        gradient = X.T @ (sigmoid(X @ beta) - y)
        beta -= eta * gradient

        if abs(np.sum(beta_old - beta)) < eps:
            print(f"Converged for i={i}")
            return beta

        beta_old = beta

    return beta


def stochastic_gradient_descent(X, y, beta, eta=1e-6, epochs=100, batch_size=100):
    """
    Stochastic gradient descent
    """
    data_indices = np.arange(X.shape[0])    # Samples
    for epoch in range(epochs):
        iter = 0
        print(f"Epoch {epoch}")
        for i in range(X.shape[0]):
            chosen_datapoints = np.random.choice(
                data_indices, size=batch_size, replace=False)

            X_sub = X[chosen_datapoints]
            y_sub = y[chosen_datapoints]

            gradient = X_sub.T @ (sigmoid(X_sub @ beta) - y_sub)

            eta = learning_schedule(epoch * X.shape[0] / batch_size + iter)

            beta -= eta * gradient

            iter += 1

        # while np.abs(np.sum(beta - beta_old)) < eps or iter < m:
        #     rand_idx = np.random.randint(int(len(X[0])))
        #     xi = X[rand_idx:rand_idx + 1]  # Matrix
        #     yi = y[rand_idx:rand_idx + 1]  # Array
        #     print(xi.shape, (xi@beta).shape, yi.shape, beta.shape)
        #     exit()
        #
        #     gradient = xi.T @ (sigmoid(xi, beta) - yi)
        #
        #     eta = learning_schedule(epoch * m + iter)
        #     beta_new = beta - eta * gradient
        #
        #     beta_old = beta
        #     beta_new = beta_old
        #
        #     iter += 1
    return beta


def accuracy(y_tilde, y):
    """ returns accruacy"""
    I = np.mean(y_tilde == y)
    return I
