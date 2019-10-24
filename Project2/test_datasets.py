from main import gradient_descent, accuracy


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


def breast_cancer():
    dataset = load_breast_cancer()

    x = dataset.data
    y = dataset.target

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


def plot_confusion_matrix(y, pred):
    conf_matrix = confusion_matrix(y, pred)
    sns.heatmap(pd.DataFrame(conf_matrix), annot=True, cmap="YlGnBu", fmt='g')
    plt.title('Confusion matrix (default = 1)')
    plt.ylabel('True value')
    plt.xlabel('Predicted value')
    plt.show()


if __name__ == "__main__":
    random_dataset()
    breast_cancer()
