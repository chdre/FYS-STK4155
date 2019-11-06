import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import scikitplot as skplt
import seaborn as sns
import matplotlib.pyplot as plt
from NeuralNetwork import *

np.random.seed(42)


def plot_heatmap(matrix, title):
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(matrix, annot=True, ax=ax, cmap="viridis")
    ax.set_title(title)
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.show()


def calculate_accuracies(NN_array, x_train, x_test, y_train, y_test, lmbda_vals, eta_vals):
    train_accuracy = np.zeros((len(eta_vals), len(lmbda_vals)))
    test_accuracy = np.zeros((len(eta_vals), len(lmbda_vals)))
    aucscore = np.zeros((len(eta_vals), len(lmbda_vals)))

    for i in range(len(eta_vals)):
        for j in range(len(lmbda_vals)):
            nn = NN_array[i][j]

            train_pred = nn.predict(x_train)
            test_pred = nn.predict(x_test)

            train_accuracy[i, j] = accuracy_score(y_train, train_pred)
            test_accuracy[i, j] = accuracy_score(y_test, test_pred)
            aucscore[i, j] = roc_auc_score(y_test, test_pred)

    return train_accuracy, test_accuracy, aucscore


def scale_date(train, test, method):
    scale = method()
    scale.fit(train)
    train = scale.transform(train)
    test = scale.transform(test)

    return train, test


def credit_card():
    from logit import data_import

    x, y, y_true = data_import()

    train_size_ = 0.7
    test_size_ = 1 - train_size_

    x_train, x_test, y_train, y_test = train_test_split(
        x, y_true, train_size=train_size_, test_size=test_size_)

    scale = StandardScaler()   # Scales by (func - mean)/std.dev
    scale.fit(x_train)
    x_train = scale.transform(x_train)
    x_test = scale.transform(x_test)

    x_train, x_test = scale_data(x_train, x_test, StandardScaler)

    epochs = 10
    batch_size = 100
    eta_vals = np.logspace(1, -7, 7)
    lmbda_vals = np.logspace(0, -7, 7)
    lmbda_vals[0] = 0

    layers = [x_train.shape[1], 100, 10, y_train.shape[1]]
    activation_func = ['sigmoid', 'sigmoid', 'sigmoid']
    if not len(layers) - 1 == len(activation_func):
        print('Add more activations functions')
        exit()

    NN_array = np.zeros((len(eta_vals), len(lmbda_vals)), dtype=object)

    # grid search
    for i, eta in enumerate(eta_vals):
        for j, lmbda in enumerate(lmbda_vals):
            NN_[i, j] = NeuralNetwork(x_train, y_train, sizes=layers,
                                      activation_function=activation_func,
                                      epochs=epochs, batch_size=batch_size, eta=eta,
                                      lmbda=lmbda).train()

    train_accuracy, test_accuracy, aucscore = calculate_accuracies(NN_array)

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
    eta_vals = np.logspace(-4, -8, 6)
    lmbda_vals = np.logspace(0, -5, 5)
    lmbda_vals[-1] = 0

    # store the models for later use
    NN_numpy = np.zeros((len(eta_vals), len(lmbda_vals)), dtype=object)

    layers = [X_train.shape[1], 50, Y_train.shape[1]]
    activation_func = ['sigmoid', 'nothing']
    if not len(layers) - 1 == len(activation_func):
        print('Add more activations functions')
        exit()

    # NN_array = np.zeros((len(eta_vals), len(lmbda_vals)), dtype=object)

    mse = np.zeros((len(eta_vals), len(lmbda_vals)))
    r2score = np.zeros((len(eta_vals), len(lmbda_vals)))

    # grid search
    for i, eta in enumerate(eta_vals):
        print(f"At {i} out of {len(eta_vals)}")
        for j, lmbda in enumerate(lmbda_vals):
            nn = NeuralNetwork(X_train, Y_train, sizes=layers, cost_function='regression',
                               activation_function=activation_func,
                               epochs=epochs, batch_size=batch_size, eta=eta,
                               lmbda=lmbda)
            nn.train()

            train_pred = nn.predict(X_train)
            test_pred = nn.predict(X_test)

            # print(train_pred.T)

            r2score[i, j] = r2_score(Y_test, test_pred)
            mse[i, j] = mean_squared_error(Y_test, test_pred)

    # train_accuracy, test_accuracy, aucscore = calculate_accuracies(
    #     NN_array, lmbda_vals, eta_vals, X_train, X_test, Y_train, Y_test)

    sns.set()
    # scikitplot.metrics.plot_cumulative_gain(Y_test, pred_test)
    # plt.show()
    plot_heatmap(r2score, 'R2 score')
    plot_heatmap(mean_squared_error, 'Mean squared error')


if __name__ == "__main__":
    # credit_card()
    # image()
    Franke_for_NN()
