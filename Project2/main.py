import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score,\
    roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
import scikitplot as skplt
import seaborn as sns
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork
from functions import *

np.random.seed(42)
plt.rcParams.update({'font.size': 12})


def neural_network_credit_card_data():
    """
    Classification of credit card default data using the neural network in
    NeuralNetwork.py. Does a gridsearch over eta and lambda, being the
    learning rate and penalty.
    Shows heatmaps of roc score, test and train accuracy as a function varying
    eta and lambda, plots of confusion matrix, roc auc and cumulative gain of
    both our neural network and for scikit-learns MLPClassifier.
    """
    x, y, y_onehot = credit_card_data_import()

    x_train, x_test, y_train, y_test, y_train_onehot, y_test_onehot = \
        train_test_split(x, y, y_onehot, test_size=0.3)

    epochs = 50
    batch_size = 100
    eta_vals = np.logspace(-7, -1, 7)
    lmbda_vals = np.logspace(-6, 1, 8)
    lmbda_vals[0] = 0  # For a zero value of penalty

    layers = [x_train.shape[1], 64, 32, 16, y_train_onehot.shape[1]]
    activation_func = ['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid']

    train_accuracy = np.zeros((len(eta_vals), len(lmbda_vals)))
    test_accuracy = np.zeros((len(eta_vals), len(lmbda_vals)))
    roc_score = np.zeros((len(eta_vals), len(lmbda_vals)))
    NN = np.zeros((len(eta_vals), len(lmbda_vals)), dtype=object)

    # grid search
    for i, eta in enumerate(eta_vals):
        for j, lmbda in enumerate(lmbda_vals):
            nn = NeuralNetwork(x_train, y_train_onehot, sizes=layers,
                               activation_function=activation_func,
                               epochs=epochs, batch_size=batch_size, eta=eta,
                               lmbda=lmbda)

            nn.train()
            NN[i, j] = nn   # Storing trained nn

            train_prob = nn.predict(x_train)
            train_pred = np.argmax(train_prob, axis=1)
            test_prob = nn.predict(x_test)
            test_pred = np.argmax(test_prob, axis=1)

            train_accuracy[i, j] = accuracy_score(y_train, train_pred)
            test_accuracy[i, j] = accuracy_score(y_test, test_pred)
            roc_score[i, j] = roc_auc_score(y_test_onehot, test_prob)

    roc_score_coord = np.argwhere(roc_score == roc_score.max())
    eta_ind = roc_score_coord[0, 0]
    lmbda_ind = roc_score_coord[0, 1]

    nn_best = NN[eta_ind, lmbda_ind]

    test_prob = nn_best.predict(x_test)
    test_pred = np.argmax(test_prob, axis=1)

    # Scikit learn neural network
    NN_sklearn = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        learning_rate="adaptive",
        activation='logistic',
        learning_rate_init=0.01,
        max_iter=1000,
        tol=1e-7,
        verbose=False)
    NN_sklearn = NN_sklearn.fit(x_train, y_train.ravel())
    test_pred_NNskl = NN_sklearn.predict(x_test)
    test_proba_NNskl = NN_sklearn.predict_proba(x_test)

    skplt.metrics.plot_confusion_matrix(y_test, test_pred, normalize=True)
    skplt.metrics.plot_confusion_matrix(y_test, test_pred_NNskl, normalize=True,
                                        title='Normalized Confusion Matrix Scikit-learn')
    skplt.metrics.plot_roc(y_test, test_prob)
    skplt.metrics.plot_roc(y_test, test_proba_NNskl,
                           title='ROC Curves Scikit-learn')
    skplt.metrics.plot_cumulative_gain(y_test, test_prob)
    skplt.metrics.plot_cumulative_gain(
        y_test, test_proba_NNskl, title='Gains Curve Scikit-learn')

    sns.set()
    plot_heatmap(train_accuracy, lmbda_vals, eta_vals, 'Training Accuracy')
    plot_heatmap(test_accuracy, lmbda_vals, eta_vals, 'Test Accuracy')
    plot_heatmap(roc_score, lmbda_vals, eta_vals, 'ROC Score')
    plt.show()


def Franke_for_NN():
    """
    Regression of credit card default data using the neural network in
    NeuralNetwork.py. Does a gridsearch over eta and lambda, being the
    learning rate and penalty.
    Shows heatmaps of R2 score and mse as a function varying
    eta and lambda. Also prints the R2 score and MSE for scikit-learns
    MLPRegressor.
    """
    n = 50
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X = np.zeros((n * n, 2))
    Y = np.zeros((n * n, 1))
    Y_true = np.zeros((n * n, 1))

    xgrid, ygrid = np.meshgrid(x, y)

    eps = np.random.normal(0, 0.5, (n, n))  # Noise

    # Dataset
    for i in range(n):
        for j in range(n):
            X[n * i + j] = [x[i], y[j]]
            FF = Franke_function(x[i], y[j])
            Y[n * i + j] = FF

    X_train, X_test, Y_train, Y_test, = train_test_split(
        X, Y, test_size=0.3)

    Y_train, Y_test = scale_data(Y_train, Y_test, StandardScaler)

    epochs = 20
    batch_size = 100
    eta_vals = np.logspace(-7, -4, 4)
    lmbda_vals = np.logspace(-4, 1, 6)
    lmbda_vals[0] = 0   # For a zero value of penalty

    layers = [X_train.shape[1], 100, 50, Y_train.shape[1]]
    activation_func = ['tanh', 'tanh', 'nothing']

    mse = np.zeros((len(eta_vals), len(lmbda_vals)))
    r2score = np.zeros((len(eta_vals), len(lmbda_vals)))
    NN = np.zeros((len(eta_vals), len(lmbda_vals)), dtype=object)

    # grid search
    for i, eta in enumerate(eta_vals):
        for j, lmbda in enumerate(lmbda_vals):
            nn = NeuralNetwork(X_train, Y_train, sizes=layers,
                               cost_function='regression',
                               activation_function=activation_func,
                               epochs=epochs, batch_size=batch_size, eta=eta,
                               lmbda=lmbda)

            nn.train()
            NN[i, j] = nn
            test_pred = nn.predict(X_test)

            r2score[i, j] = r2_score(Y_test, test_pred)
            mse[i, j] = mean_squared_error(Y_test, test_pred)

    # Scikit learn neural network
    NN_sklearn = MLPRegressor(
        hidden_layer_sizes=(100, 50),
        activation='tanh',
        learning_rate="adaptive",
        learning_rate_init=0.01,
        max_iter=1000,
        tol=1e-7,
        verbose=False)
    NN_sklearn = NN_sklearn.fit(X_train, Y_train.ravel())
    test_pred_NNskl = NN_sklearn.predict(X_test)

    print(f"R2 score {r2_score(Y_test, test_pred_NNskl)}")
    print(f"MSE {mean_squared_error(Y_test, test_pred_NNskl)}")

    sns.set()
    plot_heatmap(r2score, lmbda_vals, eta_vals, 'R2 Score')
    plot_heatmap(mse, lmbda_vals, eta_vals, "Test Accuracy")
    plt.show()


def logistic_regression_credit_card_data():
    """
    Performs a logistic regression on the credit card default data with gradient
    descent, stochastic gradient descent with mini batches.
    Performs a grid search for different learning rates. Plots the confusion
    matrix, roc auc and cumulative gain for GD, SGD and scikit-learns logistic
    regression.
    """
    x, y, y_onehot = credit_card_data_import()

    x_train, x_test, y_train, y_test, y_train_onehot, y_test_onehot = train_test_split(
        x, y, y_onehot, test_size=0.3)

    X_train = np.c_[np.array([1] * len(x_train[:, 0])), x_train]
    X_test = np.c_[np.array([1] * len(x_test[:, 0])), x_test]

    beta_init = np.random.randn(X_train.shape[1], 2)

    def calc_prob_pred(X, beta):
        "Calculates probability and prediction given X and beta."
        prob = sigmoid(X @ beta)
        pred = np.argmax(prob, axis=1)  # Returns 0 or 1 depending on max value
        return prob, pred

    beta_GD = gradient_descent(X_train, y_train_onehot, beta_init, n=10000)
    prob_GD, pred_GD = calc_prob_pred(X_test, beta_GD)

    clf = LogisticRegression(solver='lbfgs', max_iter=1e5)
    clf = clf.fit(X_train, np.ravel(y_train))
    pred_skl = clf.predict(X_test)
    prob_skl = clf.predict_proba(X_test)

    epochs = 50
    batch_size = 100
    etas = np.logspace(-6, -2, 6)

    acc_score = np.zeros(len(etas))
    roc_score = np.zeros(len(etas))

    # Grid search
    for i, eta in enumerate(etas):
        beta_SGD = stochastic_gradient_descent(
            X_train, y_train_onehot, beta_init, epochs=epochs,
            batch_size=batch_size, eta=eta)
        prob_SGD, pred_SGD = calc_prob_pred(X_test, beta_SGD)

        acc_score[i] = accuracy_score(y_test, pred_SGD)
        roc_score[i] = roc_auc_score(y_test_onehot, prob_SGD)

        if i > 0 and roc_score[i] > roc_score[i - 1]:
            best_prob_SGD, best_pred_SGD = prob_SGD, pred_SGD

    skplt.metrics.plot_confusion_matrix(
        y_test, pred_GD, normalize=True, title='Normalized Confusion Matrix (GD)')
    skplt.metrics.plot_confusion_matrix(
        y_test, best_pred_SGD, normalize=True, title='Normalized Confusion Matrix (SGD)')
    skplt.metrics.plot_confusion_matrix(
        y_test, pred_skl, normalize=True, title='Normalized Confusion Matrix (Scikit-learn)')
    skplt.metrics.plot_roc(y_test, prob_GD, title='ROC Curve (GD)')
    skplt.metrics.plot_roc(y_test, best_prob_SGD, title='ROC Curve (SGD)')
    skplt.metrics.plot_roc(y_test, prob_skl, title='ROC Curve (Scikit-learn)')

    skplt.metrics.plot_cumulative_gain(
        y_test, prob_GD, title='Gains Curve (GD)')
    skplt.metrics.plot_cumulative_gain(
        y_test, best_prob_SGD, title='Gains Curve (SGD)')
    skplt.metrics.plot_cumulative_gain(
        y_test, prob_skl, title='Gains Curve (Scikit-learn)')

    plt.show()


if __name__ == "__main__":
    neural_network_credit_card_data()
    Franke_for_NN()
    logistic_regression_credit_card_data()
