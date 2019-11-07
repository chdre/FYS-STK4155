import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score,\
    roc_auc_score
from sklearn.preprocessing import StandardScaler
import scikitplot as skplt
import seaborn as sns
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork
from functions import *

np.random.seed(42)


def neural_network_credit_card_data():
    from logit import data_import

    x, y = data_import()

    train_size_ = 0.7
    test_size_ = 1 - train_size_

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=train_size_, test_size=test_size_)

    x_train, x_test = scale_data(x_train, x_test, StandardScaler)

    epochs = 50
    batch_size = 500
    eta_vals = np.logspace(0, -7, 7)
    lmbda_vals = np.logspace(0, -7, 7)
    lmbda_vals[0] = 0

    layers = [x_train.shape[1], 100, 10, y_train.shape[1]]
    activation_func = ['tanh', 'tanh', 'tanh']
    if not len(layers) - 1 == len(activation_func):
        print('Add more activations functions')
        exit()

    train_accuracy = np.zeros((len(eta_vals), len(lmbda_vals)))
    test_accuracy_manual = np.zeros((len(eta_vals), len(lmbda_vals)))
    test_accuracy = np.zeros((len(eta_vals), len(lmbda_vals)))
    aucscore = np.zeros((len(eta_vals), len(lmbda_vals)))

    # grid search
    for i, eta in enumerate(eta_vals):
        for j, lmbda in enumerate(lmbda_vals):
            print(f"Starting for j = {j} for i = {i}")
            nn = NeuralNetwork(x_train, y_train, sizes=layers,
                               activation_function=activation_func,
                               epochs=epochs, batch_size=batch_size, eta=eta,
                               lmbda=lmbda)
            nn.train()
            train_pred = nn.predict(x_train)
            test_pred = nn.predict(x_test)

            train_accuracy[i, j] = accuracy_score(y_train, train_pred)
            test_accuracy_manual[i, j] = accuracy(y_test, test_pred)
            test_accuracy[i, j] = accuracy_score(y_test, test_pred)
            aucscore[i, j] = roc_auc_score(y_test, test_pred)

    # train_accuracy, test_accuracy, aucscore = calculate_accuracies(
    #     NN_array, x_train, x_test, y_train, y_test, lmbda_vals, eta_vals)

    sns.set()
    plot_heatmap(train_accuracy, 'Train accuracy')
    plot_heatmap(test_accuracy, 'Test accuracy')
    plot_heatmap(aucscore, 'ROC AUC score on test data')


def Franke_function(x, y):
    t1 = 0.75 * np.exp(-((9 * x - 2)**2) / 4 - ((9 * y - 2)**2) / 4)
    t2 = 0.75 * np.exp(-((9 * x + 1)**2) / 49 - ((9 * y + 1)**2) / 10)
    t3 = 0.5 * np.exp(-((9 * x - 7)**2) / 4 - ((9 * y - 3)**2) / 4)
    t4 = -0.2 * np.exp(-(9 * x - 4)**2 - (9 * y - 7)**2)

    f = t1 + t2 + t3 + t4

    return f


def Franke_for_NN():
    n = 50
    x = np.sort(np.random.rand(n))
    y = np.sort(np.random.rand(n))
    X = np.zeros((n * n, 2))
    Y = np.zeros((n * n, 1))
    Y_true = np.zeros((n * n, 1))

    eps = np.random.normal(0, 0.5, (n, n))  # Noise

    # Dataset
    for i in range(n):
        for j in range(n):
            X[n * i + j] = [x[i], y[j]]
            FF = Franke_function(x[i], y[j])
            Y[n * i + j] = FF + eps[i, j]
            Y_true[n * i + j] = FF

    train_size_ = 0.8
    test_size_ = 1 - train_size_

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, train_size=train_size_, test_size=test_size_)
    _, _, Y_true_train, Y_true_test = train_test_split(
        X, Y_true, train_size=train_size_, test_size=test_size_)

    X_train, X_test = scale_data(X_train, X_test, StandardScaler)
    Y_train, Y_test = scale_data(Y_train, Y_test, StandardScaler)
    Y_true_train, Y_true_test = scale_data(
        Y_true_train, Y_true_test, StandardScaler)

    epochs = 50
    batch_size = 200
    eta_vals = np.logspace(-2, -8, 6)
    lmbda_vals = np.logspace(0, -5, 5)
    lmbda_vals[-1] = 0

    # store the models for later use
    NN_numpy = np.zeros((len(eta_vals), len(lmbda_vals)), dtype=object)

    layers = [X_train.shape[1], 50, Y_train.shape[1]]
    activation_func = ['sigmoid', 'nothing']
    if not len(layers) - 1 == len(activation_func):
        print('Add more activations functions')
        exit()

    mse = np.zeros((len(eta_vals), len(lmbda_vals)))
    r2score = np.zeros((len(eta_vals), len(lmbda_vals)))

    # grid search
    for i, eta in enumerate(eta_vals):
        print(f"At {i} out of {len(eta_vals)}")
        for j, lmbda in enumerate(lmbda_vals):
            nn = NeuralNetwork(X_train, Y_train, sizes=layers,
                               cost_function='regression',
                               activation_function=activation_func,
                               epochs=epochs, batch_size=batch_size, eta=eta,
                               lmbda=lmbda)
            nn.train()
            train_pred = nn.predict(X_train)
            test_pred = nn.predict(X_test)

            # print(train_pred.T)

            r2score[i, j] = r2_score(Y_test, test_pred)
            mse[i, j] = mean_squared_error(Y_test, test_pred)

    sns.set()
    scikitplot.metrics.plot_cumulative_gain(Y_test, test_pred)
    plt.show()
    plot_heatmap(r2score, 'R2 score')
    plot_heatmap(mse, 'Mean squared error')


def logistic_regression_credit_card_data():
    """ main """
    x, y = data_import()

    test_size = 0.3
    train_size = 1 - test_size
    xtrain, xtest, ytrain, ytest = train_test_split(
        x, y, train_size=train_size, test_size=test_size)

    # Scaling
    scale = StandardScaler()   # Scales by (func - mean)/std.dev
    scale.fit(xtrain)
    xtrain = scale.transform(xtrain)
    xtest = scale.transform(xtest)

    Xtrain = np.c_[np.array([1] * len(xtrain[:, 0])), xtrain]
    Xtest = np.c_[np.array([1] * len(xtest[:, 0])), xtest]

    beta_init = np.random.randn(Xtrain.shape[1], 1)

    beta_GD = gradient_descent(Xtrain, ytrain, beta_init, n=1)
    pred_GD = np.round(sigmoid(Xtest @ beta_GD))
    # pred_GD = (prob_GD >= 0.5).astype(int)

    beta_SGD = gradient_descent(
        Xtrain, ytrain, beta_init, epochs=50, batch_size=500, stochastic=True)
    pred_SGD = np.round(sigmoid(Xtest @ beta_SGD))

    # clf = LogisticRegression(solver='lbfgs')
    # clf_fit = clf.fit(Xtrain, ytrain.ravel())
    # pred_skl = Xtest @ clf_fit.coef_.T

    print(f"Accuracy score for own GD: {accuracy(pred_GD, ytest)}")
    print(f"Accuracy score for own SGD: {accuracy(pred_SGD, ytest)}")
    print(f"Accuracy score scikit-learn: {clf.score(Xtest, ytest)}")

    plot_confusion_matrix(ytest, pred_GD)
    plot_confusion_matrix(ytest, pred_SGD)
    plot_confusion_matrix(ytest, pred_skl)


if __name__ == "__main__":
    credit_card()
    # image()
    # Franke_for_NN()
