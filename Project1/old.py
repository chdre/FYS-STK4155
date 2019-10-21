
def main2(method):
    p_max = 12      # Degree of polynomial in beta-coefficient

    z = DataImport(20, "Norway_1arc.tif")

    nx = len(z[0,:])
    ny = len(z[:,0])

    x = np.linspace(0,1,nx)
    y = np.linspace(0,1,ny)
    z = (z - np.mean(z))/np.std(z)

    x,y = np.meshgrid(x,y)

    # z_noise = FrankeFunction(x,y,True)
    # z_true = FrankeFunction(x,y,False)

    X = CreateDesignMatrix(x,y,p_max)       # Design matrix
    z_tilde = X @ coefficients(X,z,0)  # Predicted model using ordinary least squares

    plt.figure()
    plt.imshow(np.reshape(z_tilde,(ny, nx)), cmap='gray')

    r2score = np.zeros(p_max+1)
    mse = np.zeros(p_max+1)
    trainerror = np.zeros(p_max+1)
    testerror = np.zeros(p_max+1)
    error = np.zeros(p_max+1)
    bias = np.zeros(p_max+1)
    var = np.zeros(p_max+1)

    if method == "ridge":
        for lda in np.linspace(1e-5,1e-2,3):
            for p in range(p_max+1):
                "Looping over polynomial degree 1 to p_max"
                #Cross-validation
                r2score[p], mse[p], trainerror[p], testerror[p], error[p], bias[p], \
                var[p] = CrossValidation(x, y, z, 5, p, lda, 0, method)


            plot_EBV(range(p_max+1), error, bias, var, "Ridge with $\\lambda$ = %.4f"
                                                                        %lda)
            plot_R2MSE(range(p_max+1), r2score, mse, "Ridge with $\\lambda$ = %.4f"
                                                                        %lda)
            plot_traintestError(range(0,p_max+1), trainerror, testerror,
                                            "Ridge with $\\lambda$ = %.4f" %lda)

            print("Ridge: MSE for degree %i and lambda = %f: %f" %(p, lda, mse[p]))
            print("Ridge: R2 score for degree %i and lambda = %f: %f" %(p, lda, r2score[p]))

    elif method == "lasso":
        for gma in np.linspace(1e-4,1e-2,4):
            for p in range(p_max+1):
                r2score[p], mse[p], trainerror[p], testerror[p], error[p], bias[p], \
                var[p] = CrossValidation(x, y, z ,5 ,p ,0 ,gma , method)

            plot_EBV(range(p_max+1), error, bias, var, "Lasso with $\\gamma$ = %f" %gma)
            plot_R2MSE(range(p_max+1), r2score, mse, "Lasso with $\\gamma$ = %f" %gma)
            plot_traintestError(range(0,p_max+1), trainerror, testerror,
                                                "Lasso with $\\gamma$ = %f" %gma)

            print("Lasso: MSE for degree %i and gma=%f: %f" %(p,gma,mse[p]))
            print("Lasso: R2 score for degree %i: and gma=%f: %f" %(p,gma,r2score[p]))

    elif method == "ols":
        for p in range(p_max+1):
            "Looping over polynomial degree 1 to p_max"
            # Cross-validation
            r2score[p], mse[p], trainerror[p], testerror[p], error[p], bias[p], \
            var[p] = CrossValidation(x, y, z, 5, p, 0, 0, method)

        plot_R2MSE(range(0,p_max+1), r2score, mse, "Ordinary least squares")
        plot_traintestError(range(0,p_max+1), trainerror, testerror, "Ordinary least squares")
        plot_EBV(range(0,p_max+1), error, bias, var, "Ordinary least squares")

        print("MSE for degree %i: %f" %(p,mse[p]))
        print("R2 score for degree %i: %f" %(p, r2score[p]))


    # Confidence intervals
    Var_beta = np.array([np.diag(np.linalg.pinv(X.T.dot(X)))])
    MSE = mean_squared_error(z,np.reshape(z_tilde, (ny,nx)))
    R2 = r2_score(z,np.reshape(z_tilde, (ny,nx)))

    print("MSE: %f" %MSE)
    print("R2: %f" %R2)
