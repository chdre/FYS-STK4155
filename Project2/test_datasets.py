from logit import gradient_descent, accuracy
from sklearn.datasets import make_classification, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


from NeuralNetwork import *


def random_dataset():
    x, y = make_classification(n_samples=10000)

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

    # Scaling
    scale = StandardScaler()   # Scales by (func - mean)/std.dev
    xtrain = scale.fit_transform(xtrain)
    xtest = scale.transform(xtest)

    Xtrain = np.c_[np.array([1] * len(xtrain[:, 0])), xtrain]
    Xtest = np.c_[np.array([1] * len(xtest[:, 0])), xtest]

    beta_init = np.random.randn(len(Xtrain[0]))

    beta_GD = gradient_descent(Xtrain, ytrain, beta_init, n=10000)
    prob_GD = 1 / (1 + (np.exp(-Xtest @ beta_GD)))
    pred_GD = (prob_GD >= 0.5).astype(int)

    beta_SGD = gradient_descent(
        Xtrain, ytrain, beta_init, m=300, stochastic=True)
    prob_SGD = 1 / (1 + (np.exp(-Xtest @ beta_SGD)))
    pred_SGD = (prob_SGD >= 0.5).astype(int)

    clf = LogisticRegression(solver='lbfgs')
    pred_skl = clf.fit(xtrain, ytrain)

    print(f"Accuracy score for own GD: {accuracy(pred_GD, ytest)}")
    print(f"Accuracy score for own SGD: {accuracy(pred_SGD, ytest)}")
    print(f"Accuracy score scikit-learn: {clf.score(xtest, ytest)}")

    plot_confusion_matrix(ytest, pred_GD)
    plot_confusion_matrix(ytest, pred_SGD)
    plot_confusion_matrix(ytest, pred_skl)


def breast_cancer(plot_corr=False):
    dataset = load_breast_cancer()

    x = dataset.data
    y = dataset.target

    if plot_corr:
        df = pd.DataFrame(data=dataset)
        corr = df.corr()
        sns.heatmap(corr, annot=True, fmt='.2f')
        plt.xticks(rotation=90)
        # Because turning something 360 degrees helps??? :)
        plt.yticks(rotation=360)
        plt.show()

    # Scaling
    scale = StandardScaler()   # Scales by (func - mean)/std.dev
    x = scale.fit_transform(x)
    # Xtrain = scale.fit_transform(xtrain)
    # Xtest = scale.fit_transform(xtest)

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

    Xtrain = np.c_[np.array([1] * len(xtrain[:, 0])), xtrain]
    Xtest = np.c_[np.array([1] * len(xtest[:, 0])), xtest]

    beta_init = np.random.randn(len(Xtrain[0]))

    beta_GD = gradient_descent(Xtrain, ytrain, beta_init, n=10000)
    prob_GD = 1 / (1 + (np.exp(-Xtest @ beta_GD)))
    pred_GD = (prob_GD >= 0.5).astype(int)

    beta_SGD = gradient_descent(
        Xtrain, ytrain, beta_init, m=300, stochastic=True)
    prob_SGD = 1 / (1 + (np.exp(-Xtest @ beta_SGD)))
    pred_SGD = (prob_SGD >= 0.5).astype(int)

    clf = LogisticRegression(solver='lbfgs')
    pred_skl = clf.fit(xtrain, ytrain)

    print(f"Accuracy score for own GD: {accuracy(pred_GD, ytest)}")
    print(f"Accuracy score for own SGD: {accuracy(pred_SGD, ytest)}")
    print(f"Accuracy score scikit-learn: {clf.score(xtest, ytest)}")

    plot_confusion_matrix(ytest, pred_GD)
    plot_confusion_matrix(ytest, pred_SGD)
    plot_confusion_matrix(ytest, pred_skl)


def image_of_numbers():
    def to_categorical_numpy(integer_vector):
        n_inputs = len(integer_vector)
        categories = np.max(integer_vector) + 1
        onehot_vector = np.zeros((n_inputs, categories))
        onehot_vector[range(n_inputs), integer_vector] = 1

        return onehot_vector

    # download MNIST dataset
    dataset = load_digits()

    # define inputs and labels
    inputs = dataset.images
    labels = dataset.target

    n_inputs = len(inputs)
    inputs = inputs.reshape(n_inputs, -1)

    # one-liner from scikit-learn library
    train_size = 0.7
    test_size = 1 - train_size
    X_train, X_test, Y_train, Y_test = train_test_split(inputs, labels,
                                                        train_size=train_size,
                                                        test_size=test_size)

    Y_train_onehot, Y_test_onehot = to_categorical_numpy(
        Y_train), to_categorical_numpy(Y_test)

    epochs = 100
    batch_size = 100
    eta_vals = np.logspace(-7, 1, 9)
    lmbda_vals = np.logspace(-7, 1, 9)
    neurons = [50]  # np.linspace(50, 120, 10, dtype=int)
    activation_func = ['leaky_relu', 'softmax']

    # store the models for later use
    DNN_numpy = np.zeros((len(eta_vals), len(lmbda_vals)), dtype=object)

    # grid search
    for neus in neurons:
        layers = [X_train.shape[1], neus, Y_train_onehot.shape[1]]

        for i, eta in enumerate(eta_vals):
            for j, lmbda in enumerate(lmbda_vals):
                dnn = NeuralNetwork(X_train, Y_train_onehot, sizes=layers,
                                    activation_function=activation_func,
                                    epochs=epochs, batch_size=batch_size, eta=eta,
                                    lmbda=lmbda)

                import time as tm
                start = tm.time()
                dnn.train()
                end = tm.time()

                test_predict = dnn.predict(X_test)

                DNN_numpy[i][j] = dnn

                print("Learning rate  = ", eta)
                print("Lambda = ", lmbda)
                print("Accuracy score on test set: ",
                      accuracy_score(Y_test, test_predict))
                print(f"Runtime = {end - start}")

                print()

        sns.set()

        train_accuracy = np.zeros((len(eta_vals), len(lmbda_vals)))
        test_accuracy = np.zeros((len(eta_vals), len(lmbda_vals)))

        for i in range(len(eta_vals)):
            for j in range(len(lmbda_vals)):
                dnn = DNN_numpy[i][j]

                train_pred = dnn.predict(X_train)
                test_pred = dnn.predict(X_test)

                train_accuracy[i][j] = accuracy_score(Y_train, train_pred)
                test_accuracy[i][j] = accuracy_score(Y_test, test_pred)

        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(train_accuracy, annot=True, ax=ax, cmap="viridis")
        ax.set_title(f"Training Accuracy, neurons={neus}")
        ax.set_ylabel("$\eta$")
        ax.set_xlabel("$\lambda$")
        plt.show()

        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
        ax.set_title(f"Test Accuracy, neurons={neus}")
        ax.set_ylabel("$\eta$")
        ax.set_xlabel("$\lambda$")
        plt.show()


def plot_confusion_matrix(y, pred):
    conf_matrix = confusion_matrix(y, pred)
    sns.heatmap(pd.DataFrame(conf_matrix), annot=True, cmap="YlGnBu", fmt='g')
    plt.title('Confusion matrix (default = 1)')
    plt.ylabel('True value')
    plt.xlabel('Predicted value')
    plt.show()


if __name__ == "__main__":
    # random_dataset()
    # breast_cancer(plot_corr=False)
    image_of_numbers()
