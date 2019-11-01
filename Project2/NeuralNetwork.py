import numpy as np

np.random.seed(42)


class NeuralNetwork:
    def __init__(
            self,
            x_data,
            y_data,
            sizes,  # list of shape_x, nodes, shape_y
            activation_function='sigmoid',  # Can also be list with len = layers-1
            leaky_slope=0.1,
            eta=0.1,
            lmbda=0,
            epochs=10,
            batch_size=100):
        self.x_data_full = x_data
        self.y_data_full = y_data
        np.random.shuffle(self.y_data_full)

        self.sizes = sizes
        self.n_inputs = x_data.shape[0]
        self.n_features = x_data.shape[1]
        self.layers = len(sizes)
        self.n_neurons = sizes[1:-1]

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbda = lmbda
        self.activation_func = activation_function

        self.create_bias_and_weights()

    def create_bias_and_weights(self):
        self.biases = np.asarray([np.zeros(y) + 0.01 for y in self.sizes[1:]])
        self.weights = np.asarray([np.random.randn(x, y) for x, y in zip(
            self.sizes[:-1], self.sizes[1:])])

    def backpropagation(self):
        """ backprop"""
        error = np.empty(self.layers, dtype=np.ndarray)
        self.weights_gradient = np.empty(len(self.weights), dtype=np.ndarray)
        self.biases_gradient = np.empty(len(self.biases), dtype=np.ndarray)

        error[-1] = self.probabilities - self.y_data

        self.weights_gradient[-1] = np.matmul(self.a[-1].T, error[-1])
        self.biases_gradient[-1] = np.sum(error[-1], axis=0)
        grad1 = self.biases_gradient[-1]

        for l in range(1, self.layers):
            error[-l - 1] = np.matmul(error[-l], self.weights[-l].T) * \
                self.a[-l - 1] * (1 - self.a[-l - 1])

            self.weights_gradient[-l] = np.matmul(self.a[-l - 1].T, error[-l])
            self.biases_gradient[-l] = np.sum(error[-l], axis=0)

            if self.lmbda > 0:
                self.weights_gradient[-l] += self.lmbda * self.weights[-l]

            self.weights[-l] -= self.eta * self.weights_gradient[-l]
            self.biases[-l] -= self.eta * self.biases_gradient[-l]

    def feed_forward(self):
        """
        Performs a feed forward, storing activation and z
        """
        self.z = np.empty(len(self.biases), dtype=np.ndarray)
        self.a = np.empty(len(self.biases) + 1, dtype=np.ndarray)
        self.a[0] = self.x_data

        for i in range(len(self.biases)):
            if isinstance(self.activation_func, (list, tuple, np.ndarray)):
                self.activation_function = self.activation_func[l]
            elif isinstance(self.activation_func, str):
                self.activation_function = self.activation_func
            z = np.matmul(self.a[i], self.weights[i]) + self.biases[i]
            self.z[i] = z
            self.a[i + 1] = self.forward_activation(
                z) if i < len(self.biases) - 1 else softmax(z)
        """
        Lager aldri z av siste a
        """

        self.probabilities = softmax(z)

    def forward_activation(self, x):
        if self.activation_function == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation_function == 'tanh':
            return np.tanh(x)
        elif self.activation_function == 'relu':
            return np.maximum(0, x)
        elif self.activation_function == 'leaky_relu':
            return np.maximum(self.leaky_slope * x, x)

    def predict(self, x):
        z = np.empty(len(self.biases), dtype=np.ndarray)
        a = np.empty(len(self.biases) + 1, dtype=np.ndarray)
        a[0] = x

        for i in range(len(self.biases)):
            _z = np.matmul(a[i], self.weights[i]) + self.biases[i]
            z[i] = _z
            a[i + 1] = self.forward_activation(_z) if i < len(
                self.biases) - 1 else softmax(_z)

        exp_term = np.exp(z[-1])
        probabilities = np.argmax(
            exp_term / np.sum(exp_term, axis=1, keepdims=True), axis=1)
        return probabilities

    def train(self):
        data_indices = np.arange(self.n_inputs)

        for e in range(self.epochs):
            for i in range(self.iterations):
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False)

                self.x_data = self.x_data_full[chosen_datapoints]
                self.y_data = self.y_data_full[chosen_datapoints]

                self.feed_forward()
                self.backpropagation()


def softmax(z):
    "softmax"
    exp_term = np.exp(z)
    return exp_term / np.sum(exp_term, axis=1, keepdims=True)
