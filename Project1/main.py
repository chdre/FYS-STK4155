from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import sklearn.linear_model as skl
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold


def CreateDesignMatrix(x,y,p):
    """
    Function for creating the design matrix with rows [1, x, y, x^2, y^2, xy, x].
    x and y are arrays consisting of data points. p is the degree of the
    polynomial to fit.
    """
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((p+1)*(p+2)/2)  # Number of elements in beta
    X = np.ones((N,l))  # Design matrix

    for i in range(1,p+1):
        q = int(i*(i+1)/2)
        for j in range(i+1):
            X[:,q+j] = x**(i-j)*y**j

    return X


def FrankeFunction(x,y,noise):
    t1 = 0.75*np.exp(-((9*x-2)**2)/4 - ((9*y-2)**2)/4)
    t2 = 0.75*np.exp(-((9*x+1)**2)/49 - ((9*y+1)**2)/10)
    t3 = 0.5*np.exp(-((9*x-7)**2)/4 - ((9*y-3)**2)/4)
    t4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

    eps = np.random.normal(0, 1, (len(x), len(y)))    # Noise

    if noise == True:
        f = t1 + t2 + t3 + t4 + eps
    else:
        f = t1 + t2 + t3 + t4

    return f


def Franke_plot(x,y,z):
    """
    3D plot of the Franke function. Takes array x and y, consisting of the
    data points. z is the Franke function, returned from function FrankeFunction.
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface
    surf = ax.plot_surface(x,y,z, cmap=cm.coolwarm, linewidth=0,
                            antialiased=False)

    # Customize the z axis
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to color
    fig.colorbar(surf, shrink=0.5, aspect=5)


def coefficients(X,z,lmbda):
    """
    Calculating the coeff matrix, beta. X is the design matrix, y are
    datapoints.
    """
    # Pre-calculating
    XT = X.T
    XTX = XT.dot(X)

    beta = np.linalg.pinv(XTX + lmbda*np.identity(len(XTX[0]))).dot(XT)\
            .dot(np.ravel(z))

    return beta


def CrossValidation(x,y,z_noise,k,p,lmbda,gamma, method):
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
        z_noise_train = np.ravel(z_noise[train_ind])

        # Test data
        x_test = x[test_ind]
        y_test = y[test_ind]
        z_noise_test = np.ravel(z_noise[test_ind])

        Xtrain = CreateDesignMatrix(x_train, y_train, p)
        Xtest = CreateDesignMatrix(x_test, y_test, p)

        if method != "lasso":
            beta = coefficients(Xtrain, z_noise_train, lmbda)
            z_pred_train = Xtrain @ beta
            z_pred_test = Xtest @ beta

        else:
            model_lasso = skl.Lasso(alpha = gamma, fit_intercept=False).fit(Xtrain, z_noise_train)
            z_pred_train = model_lasso.predict(Xtrain)
            z_pred_test = model_lasso.predict(Xtest)

        # true model without noise to calculate R2 and MSE
        z_true = FrankeFunction(x, y, False)
        z_true_test = np.ravel(z_true[test_ind])

        r2score[i] = r2_score(z_true_test, z_pred_test)
        mse[i] = mean_squared_error(z_true_test, z_pred_test)

        trainerror += mean_squared_error(z_noise_train, z_pred_train)
        testerror += mean_squared_error(z_noise_test, z_pred_test)

        error += np.mean((z_noise_test - z_pred_test)**2)
        bias += np.mean((z_noise_test - np.mean(z_pred_test))**2)
        var += np.mean(np.var(z_pred_test))

        i += 1

    return np.mean(r2score), np.mean(mse), trainerror/k, testerror/k, \
            error/k, np.mean(bias)/k, np.mean(var)/k


def main(method):
    n = 100     # Number of points
    p_max = 20      # Degree of polynomial in beta-coefficient

    # Make data
    np.random.seed(10)
    x = np.sort(np.random.rand(n))
    y = np.sort(np.random.rand(n))

    x, y = np.meshgrid(x,y)

    z_noise = FrankeFunction(x,y,True)
    z_true = FrankeFunction(x,y,False)

    #   Test and train data
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y,
                                                z_noise, test_size=0.2)

    X = CreateDesignMatrix(x,y,p_max)       # Design matrix
    z_tilde = X @ coefficients(X,z_noise,0)  # Predicted model using ordinary least squares

    r2score = np.zeros(p_max+1)
    mse = np.zeros(p_max+1)
    z_pred_test = np.zeros(p_max+1)
    z_pred_train = np.zeros(p_max+1)
    trainerror = np.zeros(p_max+1)
    testerror = np.zeros(p_max+1)
    error = np.zeros(p_max+1)
    bias = np.zeros(p_max+1)
    var = np.zeros(p_max+1)

    if method == "ridge":
        parr = range(0,p_max+1)
        for lda in np.linspace(0,30,3):
            for p in range(p_max+1):
                "Looping over polynomial degree 1 to p_max"
                # Cross-validation
                r2score[p], mse[p], trainerror[p], testerror[p], error[p], bias[p], \
                var[p] = CrossValidation(x, y, z_noise, 5, p, lda, 0, method)

            plot_EBV(range(p_max+1), error, bias, var, "Lambda = %.f" %lda)
            plot_R2MSE(range(p_max+1), r2score, mse, "Lambda = %.f" %lda)

    elif method == "lasso":
        for gma in np.linspace(0.1,1,3):
            for p in range(p_max+1):
                r2score[p], mse[p], trainerror[p], testerror[p], error[p], bias[p], \
                var[p] = CrossValidation(x, y, z_noise ,5 ,p ,0 , gma, method)

            plot_EBV(range(p_max+1), error, bias, var, "Gamma = %.2f" %gma)
            plot_R2MSE(range(p_max+1), r2score, mse, "Gamma = %.2f" %gma)

    elif method == "ols":
        for p in range(p_max+1):
            "Looping over polynomial degree 1 to p_max"
            # Cross-validation
            r2score[p], mse[p], trainerror[p], testerror[p], error[p], bias[p], \
            var[p] = CrossValidation(x, y, z_noise, 5, p, 0, 0, method)

        plot_R2MSE(range(0,p_max+1), r2score, mse, "PLØTT")
        plot_traintestError(range(0,p_max+1), trainerror, testerror, "polt")
        plot_EBV(range(0,p_max+1), error, bias, var, "DETTE ER PLOTTE")

    # Confidence intervals
    Var_beta = np.array([np.diag(np.linalg.pinv(X.T.dot(X)))])
    MSE = mean_squared_error(z_true,np.reshape(z_tilde,(n,n)))
    R2 = r2_score(z_true, np.reshape(z_tilde,(n,n)))

    # Franke_plot(x,y,z)
    # Franke_plot(x,y,np.reshape(z_tilde, (n,n)))
    # Franke_plot(x_test,y_test,np.reshape(z_tilde_CV, (n,n)))


def plot_traintestError(range,trainerror, testerror, title):
    plt.figure()
    plt.title(title)
    plt.plot(range, trainerror, label="Train error")
    plt.plot(range, testerror, label="Test error")
    plt.legend()


def plot_EBV(range,error,bias,var,title):
    plt.figure()
    plt.title(title)
    plt.plot(range, error, label="Error")
    plt.plot(range, bias, label="Bias")
    plt.plot(range, var, label="Variance")
    plt.legend()


def plot_R2MSE(range, r2score, mse, title):
    plt.figure()
    plt.title(title)
    plt.plot(range, r2score, "-", label="R2 Score")
    plt.plot(range, mse, label="MSE")
    plt.legend()


if __name__ == "__main__":
    main("lasso")
    plt.show()
