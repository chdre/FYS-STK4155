from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import sklearn.linear_model as skl
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold
from imageio import imread

plt.rcParams.update({'font.size': 12})


def CreateDesignMatrix(x, y, p):
    """
    Function for creating the design matrix with rows [1, x, y, x^2, y^2, xy, x].
    x and y are arrays consisting of data points. p is the degree of the
    polynomial to fit.
    """
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((p + 1) * (p + 2) / 2)  # Number of elements in beta
    X = np.ones((N, l))  # Design matrix

    for i in range(1, p + 1):
        q = int(i * (i + 1) / 2)
        for j in range(i + 1):
            X[:, q + j] = x**(i - j) * y**j

    return X


def coefficients(X, z, lmbda):
    """
    Calculating the coeff matrix, beta. X is the design matrix, y are
    datapoints.
    """
    # Pre-calculating
    XT = X.T
    XTX = XT.dot(X)

    beta = np.linalg.pinv(XTX + lmbda * np.identity(len(XTX[0]))).dot(XT)\
        .dot(np.ravel(z))

    return beta


def FrankeFunction(x, y, noise, scale):
    t1 = 0.75 * np.exp(-((9 * x - 2)**2) / 4 - ((9 * y - 2)**2) / 4)
    t2 = 0.75 * np.exp(-((9 * x + 1)**2) / 49 - ((9 * y + 1)**2) / 10)
    t3 = 0.5 * np.exp(-((9 * x - 7)**2) / 4 - ((9 * y - 3)**2) / 4)
    t4 = -0.2 * np.exp(-(9 * x - 4)**2 - (9 * y - 7)**2)

    eps = np.random.normal(0, scale, (len(x), len(y)))    # Noise

    if noise == True:
        f = t1 + t2 + t3 + t4 + eps
    else:
        f = t1 + t2 + t3 + t4

    return f


def Franke_data(n, scale):
    np.random.seed(10)
    x = np.sort(np.random.rand(n))
    y = np.sort(np.random.rand(n))
    x, y = np.meshgrid(x, y)

    z = FrankeFunction(x, y, True, scale)    # With noise
    z_true = FrankeFunction(x, y, False, 0)

    return z, z_true, x, y


def terrain_data(sc, filename):
    z = DataImport(10, "Norway_1arc.tif")

    nx = len(z[0, :])
    ny = len(z[:, 0])

    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    z = (z - np.mean(z)) / np.std(z)

    x, y = np.meshgrid(x, y)

    return z, x, y


def DataImport(sc, filename):
    """
    Imports terraindata. sc: scaling. Returns array of downscaled image.
    """
    # Load the terrain
    terrain1 = imread(filename)

    # Scale the terrain to reduce size of array
    downscaled = terrain1[1::sc, 1::sc]

    # Show the terrain
    # plt.figure()
    # plt.imshow(terrain1, cmap='gray')
    # plt.figure()
    # plt.imshow(downscaled, cmap='gray')

    return downscaled


def CrossValidation(x, y, z, k, p, lmbda, gamma, method, scale):
    """
    Performs a k-fold crossvalidation of
    x and y are data, z is the function we want to approximate. k is the number
    of folds, p_max is the maximum polynomial degree in OLS we want to test for.
    Returns the best beta.

    For each pass of CV, calculate MSE and R2 score. For each pass of p store
    the mean MSE and R2 score into array. Returns two arrays containing mean
    MSE and R2 score as we increase complexity
    """
    kf = KFold(k, True, 1)  # Settings indexes for train and test for k folds

    r2score = np.zeros(k)
    mse = np.zeros(k)

    i = trainerror = testerror = error = bias = var = 0

    for train_ind, test_ind in kf.split(x):
        "Looping over the train and test indices of the k-folds"
        # Train data
        x_train = x[train_ind]
        y_train = y[train_ind]
        z_train = np.ravel(z[train_ind])

        # Test data
        x_test = x[test_ind]
        y_test = y[test_ind]
        z_test = np.ravel(z[test_ind])

        Xtrain = CreateDesignMatrix(x_train, y_train, p)
        Xtest = CreateDesignMatrix(x_test, y_test, p)

        if method == "ridge":
            beta = coefficients(Xtrain, z_train, lmbda)
            z_pred_train = Xtrain @ beta
            z_pred_test = Xtest @ beta

        if method == "lasso":
            model_lasso = skl.Lasso(alpha=gamma, fit_intercept=False,
                                    max_iter=3e4, tol=0.01).fit(Xtrain, z_train)
            z_pred_train = model_lasso.predict(Xtrain)
            z_pred_test = model_lasso.predict(Xtest)

        if method == "ols":
            # true model without noise to calculate R2 and MSE
            beta = coefficients(Xtrain, z_train, 0)

            z = FrankeFunction(x, y, False, 0)
            z_test = np.ravel(z[test_ind])
            z_train = np.ravel(z[train_ind])

            z_pred_train = Xtrain @ beta
            z_pred_test = Xtest @ beta

        r2score[i] = r2_score(z_test, z_pred_test)
        mse[i] = mean_squared_error(z_test, z_pred_test)

        trainerror += mean_squared_error(z_train, z_pred_train)
        testerror += mean_squared_error(z_test, z_pred_test)

        error += np.mean((z_test - z_pred_test)**2)
        bias += np.mean((z_test - np.mean(z_pred_test))**2)
        var += np.mean(np.var(z_pred_test))

        i += 1

    return np.mean(r2score), np.mean(mse), trainerror / k, testerror / k, \
        np.mean(error) / k, np.mean(bias) / k, np.mean(var) / k


def plot_traintestError(range, trainerror, testerror, title):
    plt.figure()
    plt.title(title)
    plt.plot(range, trainerror, label="Train error")
    plt.plot(range, testerror, label="Test error")
    plt.xlabel("Complexity")
    plt.ylabel("Error")
    plt.legend()


def plot_ErrBiasVar(range, error, bias, var, title):
    plt.figure()
    plt.title(title)
    plt.plot(range, error, label="Error")
    plt.plot(range, bias, label="Bias")
    plt.plot(range, var, label="Variance")
    plt.xlabel("Complexity")
    plt.legend()


def plot_R2MSE(range, r2score, mse, title):
    plt.figure()
    plt.title(title)
    plt.plot(range, r2score, "-", label="R2 Score")
    plt.plot(range, mse, label="MSE")
    plt.xlabel("Complexity")
    plt.legend()


def Franke_plot(x, y, z):
    """
    3D plot of the Franke function. Takes array x and y, consisting of the
    data points. z is the Franke function, returned from function FrankeFunction.
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0,
                           antialiased=False)

    # Customize the z axis
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to color
    fig.colorbar(surf, shrink=0.5, aspect=5)
