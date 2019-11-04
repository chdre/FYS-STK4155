from functions import *


def main(method, data):
    p_max = 20      # Degree of polynomial in beta-coefficient
    scale = 0.5     # Scale of error in Franke function

    if data == FrankeFunction:
        n = 50
        z, z_true, x, y = Franke_data(n, scale)
        # =========== First task ===========
        # OLS for p = 0
        X = CreateDesignMatrix(x, y, p_max)       # Design matrix
        # Predicted model using ordinary least squares
        z_tilde = X @ coefficients(X, z, 0)

        Var_beta = np.array([np.diag(np.linalg.pinv(X.T.dot(X)))])
        MSE = mean_squared_error(z_true, np.reshape(z_tilde, (n, n)))
        R2 = r2_score(z_true, np.reshape(z_tilde, (n, n)))
        # ===========

    else:
        z, x, y = terrain_data(10, "Norway_1arc.tif")

    #   Test and train data
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y,
                                                                         z, test_size=0.2)

    r2score = np.zeros(p_max + 1)
    mse = np.zeros(p_max + 1)
    trainerror = np.zeros(p_max + 1)
    testerror = np.zeros(p_max + 1)
    error = np.zeros(p_max + 1)
    bias = np.zeros(p_max + 1)
    var = np.zeros(p_max + 1)

    if method == "ridge":
        parr = range(0, p_max + 1)
        for lda in np.linspace(1e-4, 1e-2, 3):
            for p in range(p_max + 1):
                "Looping over polynomial degree 1 to p_max"
                # Cross-validation
                r2score[p], mse[p], trainerror[p], testerror[p], error[p], bias[p], \
                    var[p] = CrossValidation(x, y, z, 5, p, lda, 0, method, 0)

            plot_ErrBiasVar(range(p_max + 1), error, bias, var,
                            "Ridge with $\\lambda$ = %f" % lda)
            plot_traintestError(range(0, p_max + 1), trainerror, testerror,
                                "Ridge with $\\lambda$ = %f" % lda)
            plot_R2MSE(range(p_max + 1), r2score, mse, "Ridge with $\\lambda$ = %f"
                       % lda)

    elif method == "lasso":
        for gma in np.linspace(1e-4, 1e-2, 5):
            for p in range(p_max + 1):
                r2score[p], mse[p], trainerror[p], testerror[p], error[p], bias[p], \
                    var[p] = CrossValidation(x, y, z, 5, p, 0, gma, method, 0)

            plot_ErrBiasVar(range(p_max + 1), error, bias, var,
                            "Lasso with $\\gamma$ = %f" % gma)
            plot_traintestError(range(p_max + 1), trainerror, testerror,
                                "Lasso with $\\gamma$ = %f" % gma)
            plot_R2MSE(range(p_max + 1), r2score, mse, "Lasso with $\\gamma$ = %f"
                       % gma)

    elif method == "ols":
        for p in range(p_max + 1):
            "Looping over polynomial degree 1 to p_max"
            # Cross-validation
            r2score[p], mse[p], trainerror[p], testerror[p], error[p], bias[p], \
                var[p] = CrossValidation(x, y, z, 5, p, 0, 0, method, scale)

        plot_R2MSE(range(p_max + 1), r2score, mse, "Ordinary least squares")
        plot_traintestError(range(p_max + 1), trainerror,
                            testerror, "Ordinary least squares")
        plot_ErrBiasVar(range(p_max + 1), error, bias,
                        var, "Ordinary least squares")


if __name__ == "__main__":
    # main2("lasso")
    main("ols", FrankeFunction)
    # DataImport("Norway_1arc.tif")
    plt.show()
