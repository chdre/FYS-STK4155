import numpy as np
import sklearn.linear_model as skl
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt


def DM(x, n):
    """
    Function that creates the design matrix of argument x, which is an array.
    Argument n is the polynomial degree.
    """
    X = np.ones((len(x),n+1))

    for i in range(1,n+1):
        X[:,i] = x[:,0]**i

    return X

def coeffs(X, y):
    """
    Calculating the coefficient matrix beta. X is the design matrix, y is a
    function.
    """
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    return beta

def lin_reg_skl(X,y):
    """
    Does a linear regression using design matrix X and y, with scikit learning.
    Returns the prediction y_tilde.
    """
    clf = skl.LinearRegression().fit(X,y)
    y_tilde = clf.predict(X)

    return y_tilde

def main(n):
    x = np.random.rand(100)
    y = 5*x*x + 0.1*np.random.randn(100)

    X = DM(x,n) # Design matrix

    y_tilde = X @ coeffs(X,y) # Prediction
    y_tilde_skl = lin_reg_skl(X,y)

    MSE = mean_squared_error(y, y_tilde)
    R2_score = r2_score(y, y_tilde)
    print(MSE, R2_score)

    # plt.plot(y_tilde)
    # plt.plot(y_tilde_skl)
    # plt.plot(y)
    # plt.show()


if __name__ == "__main__":
    main(2) #For second order polynomial
