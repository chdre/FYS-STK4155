import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from task2 import DM


def coeffs_ridge(X, y, _lambda, n):
    """
    X is the design matrix. y are the data we wish to fit. lambda is a shrinkage
    factor of the coefficients.
    Calculates the coefficients for Ridge regression. Returns the coefficients.
    """
    beta = np.linalg.inv(X.T.dot(X) + _lambda*np.identity(n+1)).dot(X.T).dot(y)

    return beta

def main(n, _lambda):
    np.random.seed(0)
    x = np.random.rand(100,1)
    y = 5*x*x + 0.1*np.random.randn(100,1)


    X = DM(x,n)

    y_tilde = X @ coeffs_ridge(X, y, _lambda, n)
    clf = skl.Ridge(alpha = _lambda).fit(x,y)
    betaa = clf.coef_

    # print(betaa)
    print(coeffs_ridge(X, y, _lambda, n))

    y_tilde_ridge = clf.predict(x)

    # plt.figure()
    # plt.plot(y_tilde_ridge)
    # plt.plot(y)
    # plt.legend([_lambda])

if __name__ == "__main__":
    # main(2, 10)
    main(2,0.1)
    # main(2,0.0001)
    plt.show()
