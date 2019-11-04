import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from NeuralNetwork import *

np.random.seed(42)


def credit_card():
    from logit import data_import

    x, y, y_true = data_import()

    train_size_ = 0.7
    test_size_ = 1 - train_size_

    x_train, x_test, y_train, y_test = train_test_split(
        x, y_true, train_size=train_size_, test_size=test_size_)
    _, _, y_train_onehot, y_test_onehot = train_test_split(
        x, y, test_size=test_size_)

    scale = StandardScaler()   # Scales by (func - mean)/std.dev
    scale.fit(x_train)
    x_train = scale.transform(x_train)
    x_test = scale.transform(x_test)

    epochs = 20
    batch_size = 2000
    hidden_neurons = 100
    eta_vals = np.logspace(-7, 1, 7)
    lmbda_vals = np.logspace(-7, 1, 7)
    lmbda_vals[0] = 0
    # store the models for later use
    DNN_numpy = np.zeros((len(eta_vals), len(lmbda_vals)), dtype=object)
    layers = [len(x_train[0]), 48, len(y_train_onehot[0])]
    activation_func = ['sigmoid', 'softmax']

    # grid search
    for i, eta in enumerate(eta_vals):
        for j, lmbda in enumerate(lmbda_vals):
            dnn = NeuralNetwork(x_train, y_train_onehot,
                                activation_function=activation_func, sizes=layers,
                                epochs=epochs, batch_size=batch_size, eta=eta,
                                lmbda=lmbda)
            dnn.train()
            test_predict = dnn.predict(x_test)

            DNN_numpy[i][j] = dnn

            print("Learning rate  = ", eta)
            print("Lambda = ", lmbda)
            print("Accuracy score on test set: ",
                  accuracy_score(y_test, test_predict))
            print()

    sns.set()

    train_accuracy = np.zeros((len(eta_vals), len(lmbda_vals)))
    test_accuracy = np.zeros((len(eta_vals), len(lmbda_vals)))

    for i in range(len(eta_vals)):
        for j in range(len(lmbda_vals)):
            dnn = DNN_numpy[i][j]

            train_pred = dnn.predict(x_train)
            test_pred = dnn.predict(x_test)

            train_accuracy[i][j] = accuracy_score(y_train, train_pred)
            test_accuracy[i][j] = accuracy_score(y_test, test_pred)

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(train_accuracy, annot=True, ax=ax, cmap="viridis")
    ax.set_title("Training Accuracy")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
    ax.set_title("Test Accuracy")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.show()


def image():
    from test_datasets import image_of_numbers

    image_of_numbers()


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

    eps = np.random.normal(0, 0.01, (n, n))  # Noise

    for i in range(n):
        for j in range(n):
            X[n * i + j] = [x[i], y[j]]
            Y[n * i + j] = Franke_function(x[i], y[j]) + eps[i, j]

    train_size_ = 0.8
    test_size_ = 1 - train_size_

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, train_size=train_size_, test_size=test_size_)

    scale = StandardScaler()
    scale.fit(X_train)
    X_train = scale.transform(X_train)
    X_test = scale.transform(X_test)
    scale.fit(Y_train)
    Y_train = scale.transform(Y_train)
    Y_test = scale.transform(Y_test)

    layers = [X_train.shape[1], 50, Y_train.shape[1]]

    nn = NeuralNetwork(X_train, Y_train, layers,
                       activation_function=['sigmoid', 'sigmoid'])

    nn.train()

    pred_test = nn.predict(X_test)

    # print("Learning rate  = ", eta)
    # print("Lambda = ", lmbda)
    r2score = r2_score(Y_test, pred_test)
    mse = mean_squared_error(Y_test, pred_test)

    print(f"R2 score = {r2_score}")
    print(f"MSE = {mse}")


if __name__ == "__main__":
    # credit_card()
    # image()
    Franke_for_NN()
