import numpy as np
from main import data_import
from sklearn.model_selection import train_test_split


class NeuralNetwork:
    def __init__(
        self,
        x_data,
        y_data,
        sizes,  # list of shape_x, nodes, shape_y
        eta=0.1,
        epochs=10,
        batch_size=100,
    ):
        self.x_data_full = x_data
        self.y_data_full = y_data

        self.sizes = sizes
        self.n_inputs = sizes[0]
        self.n_features = sizes[-1]
        self.layers = len(sizes)
        self.n_neurons = sizes[1:-1]

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta

        self.create_bias_and_weights()

    def create_bias_and_weights(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(x, y) for x, y in zip(
            self.sizes[:-1], self.sizes[1:])]

    def feed_forward(self):
        """
        Performs a feed forward, storing activation and z
        """
        self.z = []
        self.a = [self.x_data]
        self.probabilities = []
        a_temp = self.x_data

        print("hererer")

        for b, w in zip(self.biases, self.weights):
            print("here")
            _z = np.matmul(a_temp, w) + b
            self.z.append(_z)
            a_temp = sigmoid(_z)   # Updating
            self.a.append(a_temp)

        print("hererr")

        # Softmax
        exp_term = np.exp(self.z[-1])    # Temporary to reduce FLOPS
        self.probabilities = exp_term / \
            np.sum(exp_term, axis=1, keepdims=True)

    def backpropagation(self):
        """ backprop"""
        error_output = [p - self.y_data for p in self.probabilities]
        error_hidden = [np.matmul(error_output, w.T) * a * (1 - a)
                        for w, a in zip(self.weights, self.a)]

        self.weights_gradient = [np.zeros(w.shape) for w in self.weights]
        self.biases_gradient = [np.zeros(b.shape) for b in self.biases]

        "Check if error_hidden indices are correct"
        self.weights_gradient[-1] = np.matmul(error_out[-1], a[-2].T)
        self.biases_gradient[-1] = error_hidden[-1]

        for l in range(2, self.hidden_layers):
            z = self.z[-l]
            temp = np.matmul(self.weights[-l + 1].T, error_hidden[-l + 1])
            self.weights_gradient[-l] = temp
            self.biases_gradient[-l] = np.matmul(temp, a[-l - 1].T)

        self.weights -= self.eta * self.weights_gradient
        self.bias -= self.eta * self.biases_gradient

    def predict(self, x):
        probabilites = self.feed_forward(x)
        return

    def train(self):
        data_indices = np.arange(self.n_inputs)

        for e in range(self.epochs):
            for i in range(self.iterations):
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False
                )

                self.x_data = self.x_data_full[chosen_datapoints]
                self.y_data = self.y_data_full[chosen_datapoints]

                self.feed_forward()
                self.backpropagation()


def sigmoid(z):
    """ sigmoid =)))"""
    return 1 / (1 + np.exp(-z))


def main():
    x, y = data_import()

    test_size_ = 0.3
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size_)

    _sizes = [len(x_train[:, 0]), 50, len(y_train)]

    dnn = NeuralNetwork(x_train, y_train, sizes=_sizes)

    dnn.train()

    test_predict = dnn.predict(x_test)


if __name__ == "__main__":
    main()
