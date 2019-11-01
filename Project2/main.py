import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from NeuralNetwork import *


def credit_card():
    from logit import data_import

    x, y, y_true = data_import()

    train_size_ = 0.7
    test_size_ = 1 - train_size_

    x_train, x_test, y_train, y_test = train_test_split(
        x, y_true, train_size=train_size_, test_size=test_size_)
    _, _, y_train_onehot, y_test_onehot = train_test_split(
        x, y, test_size=test_size_)

    epochs = 20
    batch_size = 2000
    hidden_neurons = 100
    eta_vals = np.logspace(-5, 1, 7)
    lmbda_vals = np.logspace(-5, 1, 7)
    # store the models for later use
    DNN_numpy = np.zeros((len(eta_vals), len(lmbda_vals)), dtype=object)
    layers = [len(x_train[0]), 50, len(y_train_onehot[0])]
    activation_func = 'sigmoid'

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


if __name__ == "__main__":
    credit_card()
    # image()
