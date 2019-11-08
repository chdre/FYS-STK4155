import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score,\
    roc_auc_score
from sklearn.linear_model import LogisticRegression
import scikitplot as skplt
import seaborn as sns
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork
from functions import *
from matplotlib import cm

np.random.seed(42)


def neural_network_credit_card_data():
    x, y, y_onehot = credit_card_data_import()

    x_train, x_test, y_train, y_test, y_train_onehot, y_test_onehot = \
        train_test_split(x, y, y_onehot, test_size=0.3)

    epochs = 1
    batch_size = 100
    eta_vals = np.logspace(0, -7, 7)
    lmbda_vals = np.logspace(0, -7, 7)
    lmbda_vals[-1] = 0

    layers = [x_train.shape[1], 100, y_train_onehot.shape[1]]
    activation_func = ['sigmoid', 'sigmoid']
    if not len(layers) - 1 == len(activation_func):
        print('Add more activations functions')
        exit()

    train_accuracy = np.zeros((len(eta_vals), len(lmbda_vals)))
    test_accuracy_manual = np.zeros((len(eta_vals), len(lmbda_vals)))
    test_accuracy = np.zeros((len(eta_vals), len(lmbda_vals)))
    auc_score = np.zeros((len(eta_vals), len(lmbda_vals)))
    NN = np.zeros((len(eta_vals), len(lmbda_vals)), dtype=object)

    # grid search
    for i, eta in enumerate(eta_vals):
        for j, lmbda in enumerate(lmbda_vals):
            print(f"Starting for j = {j} for i = {i}")
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
            test_accuracy_manual[i, j] = accuracy(y_test, test_pred)
            test_accuracy[i, j] = accuracy_score(y_test, test_pred)
            auc_score[i, j] = roc_auc_score(y_test, np.max(test_prob, axis=1))

    auc_score_coord = np.argwhere(auc_score == auc_score.max())
    eta_ind = auc_score_coord[0, 0]
    lmbda_ind = auc_score_coord[0, 1]

    nn_best = NN[eta_ind, lmbda_ind]

    test_prob = nn_best.predict(x_test)
    test_pred = np.argmax(test_prob, axis=1)

    skplt.metrics.plot_confusion_matrix(y_test, test_pred, normalize=True)
    skplt.metrics.plot_roc(y_test, test_prob)
    skplt.metrics.plot_cumulative_gain(y_test, test_prob)

    sns.set()
    plot_heatmap(train_accuracy, 'Train accuracy', lmbda_vals, eta_vals)
    plot_heatmap(test_accuracy, 'Test accuracy', lmbda_vals, eta_vals)
    plot_heatmap(auc_score, 'ROC AUC score on test data', lmbda_vals, eta_vals)
    plt.show()


def Franke_for_NN(plot_heatmap=True):
    n = 50
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
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

    X_train, X_test, Y_train, Y_test, Y_true_train, Y_true_test = train_test_split(
        X, Y, Y_true, test_size=0.3)

    X_train, X_test = scale_data(X_train, X_test, StandardScaler)
    Y_train, Y_test = scale_data(Y_train, Y_test, StandardScaler)
    Y_true_train, Y_true_test = scale_data(
        Y_true_train, Y_true_test, StandardScaler)

    epochs = 1
    batch_size = 100
    eta_vals = np.logspace(-2, -8, 6)
    lmbda_vals = np.logspace(0, -5, 5)
    lmbda_vals[-1] = 0

    layers = [X_train.shape[1], 25, Y_train.shape[1]]
    activation_func = ['sigmoid', 'nothing']
    if not len(layers) - 1 == len(activation_func):
        print('Add more activations functions')
        exit()

    mse = np.zeros((len(eta_vals), len(lmbda_vals)))
    r2score = np.zeros((len(eta_vals), len(lmbda_vals)))
    NN = np.zeros((len(eta_vals), len(lmbda_vals)), dtype=object)

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
            NN[i, j] = nn
            test_pred = nn.predict(X_test)

            r2score[i, j] = r2_score(Y_true_test, test_pred)
            mse[i, j] = mean_squared_error(Y_true_test, test_pred)

    # sns.set()
    # plot_heatmap(r2score, 'R2 score', lmbda_vals, eta_vals)
    # plot_heatmap(mse, 'Mean squared error', lmbda_vals, eta_vals)
    # plt.show()


def logistic_regression_credit_card_data():
    """ main """
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

    # beta_SGD = stochastic_gradient_descent(
    #     X_train, y_train_onehot, beta_init, epochs=20, batch_size=100, mini_batches=False)
    # prob_SGD, pred_SGD = calc_prob_pred(X_test, beta_SGD)

    clf = LogisticRegression(solver='lbfgs', max_iter=1e5)
    clf = clf.fit(X_train, np.ravel(y_train))
    pred_skl = clf.predict(X_test)
    prob_skl = clf.predict_proba(X_test)

    etas = np.logspace(1, -7, 7)
    logreg_array = np.zeros(len(etas))

    prob_SGD_arr = np.zeros(len(etas))
    pred_SGD_arr = np.zeros(len(etas))
    acc_score = np.zeros(len(etas))
    roc_score = np.zeros(len(etas))

    # Grid search
    for i, eta in enumerate(etas):
        beta_SGD = stochastic_gradient_descent(
            X_train, y_train_onehot, beta_init, epochs=20, batch_size=100, eta=eta)
        prob_SGD[i], pred_SGD[i] = calc_prob_pred(X_test, beta_SGD)

        acc_score[i] = accuracy_score(y_test, pred_SGD)
        roc_score[i] = roc_auc_score(y_test, prob_SGD)

        if i > 0 and roc_score[i] > roc_score[i - 1]:
            best_prob_SGD, best_pred_SGD = prob_SGD[i], pred_SGD[i]

    best_eta = etas(np.argmax(roc_score))

    skplt.metrics.plot_confusion_matrix(y_test, pred_GD, normalize=True)
    skplt.metrics.plot_confusion_matrix(y_test, pred_SGD, normalize=True)
    skplt.metrics.plot_confusion_matrix(y_test, pred_skl, normalize=True)
    skplt.metrics.plot_roc(y_test, prob_GD)
    skplt.metrics.plot_roc(y_test, prob_SGD)
    skplt.metrics.plot_roc(y_test, prob_skl)
    skplt.metrics.plot_cumulative_gain(Y_test, prob_GD)
    skplt.metrics.plot_cumulative_gain(Y_test, prob_SGD)
    skplt.metrics.plot_cumulative_gain(Y_test, prob_skl)

    plt.show()


if __name__ == "__main__":
    # neural_network_credit_card_data()
    Franke_for_NN()
    # logistic_regression_credit_card_data()
