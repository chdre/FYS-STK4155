import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score,\
    roc_auc_score
from sklearn.linear_model import LogisticRegression
import scikitplot as skplt
import seaborn as sns
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork
from functions import *

np.random.seed(42)


def neural_network_credit_card_data():
    x, y, y_onehot = credit_card_data_import()

    x_train, x_test, y_train, y_test, y_train_onehot, y_test_onehot = \
        train_test_split(x, y, y_onehot, test_size=0.3)

    epochs = 50
    batch_size = 500
    eta_vals = np.logspace(0, -7, 7)
    lmbda_vals = np.logspace(0, -7, 7)
    lmbda_vals[-1] = 0

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
            train_prob = nn.predict(x_train)
            test_prob = nn.predict(x_test)

            train_accuracy[i, j] = accuracy_score(
                y_train, np.round(train_pred))
            test_accuracy_manual[i, j] = accuracy(y_test, np.round(test_pred))
            test_accuracy[i, j] = accuracy_score(y_test, np.round(test_pred))
            aucscore[i, j] = roc_auc_score(y_test, test_pred)

    # train_accuracy, test_accuracy, aucscore = calculate_accuracies(
    #     NN_array, x_train, x_test, y_train, y_test, lmbda_vals, eta_vals)

    sns.set()
    plot_heatmap(train_accuracy, 'Train accuracy', lmbda_vals, eta_vals)
    plot_heatmap(test_accuracy, 'Test accuracy', lmbda_vals, eta_vals)
    plot_heatmap(aucscore, 'ROC AUC score on test data', lmbda_vals, eta_vals)


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
    x, y, y_onehot = credit_card_data_import()

    x_train, x_test, y_train, y_test, y_train_onehot, y_test_onehot = train_test_split(
        x, y, y_onehot, test_size=0.3)

    X_train = np.c_[np.array([1] * len(x_train[:, 0])), x_train]
    X_test = np.c_[np.array([1] * len(x_test[:, 0])), x_test]

    beta_init = np.random.randn(X_train.shape[1], 2)

    def calc_prob_pred(X, beta):
        "Calculates probability and prediction given X and beta."
        prob = sigmoid(X @ beta)
        pred = np.argmax(np.round(prob), axis=1)
        return prob, pred

    # beta_GD = gradient_descent(X_train, y_train_onehot, beta_init, n=10000)
    # prob_GD, pred_GD = calc_prob_pred(X_test, beta_GD)
    #
    # beta_SGD = stochastic_gradient_descent(
    #     X_train, y_train_onehot, beta_init, epochs=20, batch_size=100)
    # prob_SGD, pred_SGD = calc_prob_pred(X_test, beta_SGD)

    clf = LogisticRegression(solver='lbfgs', max_iter=1e5)
    clf = clf.fit(X_train, np.ravel(y_train))
    pred_skl = clf.predict(X_test)
    prob_skl = clf.predict_proba(X_test)

    etas = np.logspace(1, -7, 7)
    logreg_array = np.zeros(len(etas))

    # Grid search
    for eta in etas:
        beta_SGD = stochastic_gradient_descent(
            X_train, y_train_onehot, beta_init, epochs=20, batch_size=100, eta=eta)
        temp_prob_SGD, temp_pred_SGD = calc_prob_pred(X_test, beta_SGD)
        acc_score = accuracy_score(y_test, temp_pred_SGD)
        roc_score = roc_auc_score(y_test, temp_prob_SGD)
        if acc_score > acc_score_prev and roc_score > roc_score_prev:
            prob_SGD, pred_SGD = temp_prob_SGD, temp_pred_SGD
            best_eta = eta

    print(best_eta)

    # skplt.metrics.plot_confusion_matrix(y_test, pred_GD, normalize=True)
    skplt.metrics.plot_confusion_matrix(y_test, pred_SGD, normalize=True)
    skplt.metrics.plot_confusion_matrix(y_test, pred_skl, normalize=True)
    # skplt.metrics.plot_roc(y_test, prob_GD)
    skplt.metrics.plot_roc(y_test, prob_SGD)
    skplt.metrics.plot_roc(y_test, prob_skl)

    plt.show()


if __name__ == "__main__":
    # neural_network_credit_card_data()
    # Franke_for_NN()
    logistic_regression_credit_card_data()
