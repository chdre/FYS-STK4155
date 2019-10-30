class NeuralNetwork:
    def __init__(
        self,
        x_data,
        y_data,
        eta=0.1,
        n_neurons=50,  # Can be a list/array, each element => hidden layer
        hidden_layers=len(n_hidden_neurons),
        epochs=10,
        batch_size=100,
    ):
        self.x_data = x_data
        self.y_data = y_data

        self.n_inputs = x_data.shape[0]
        self.n_features = x_data.shape[1]
        self.hiddel_layers = hidden_layers
        self.n_neurons = n_neurons

        self.epochs = epochs
        self.batch_size = batch_size
        self.iteration = self.n_inputs // self.batch_size
        self.eta = eta

        self.create_bias_weights()

    def create_bias_and_weights(self):
        self.biases = np.array([np.random.randn(y, 1) for y in n_neurons[1:]])
        self.weights = np.array([np.random.randn(x, y)
                                 for x, y in zip(n_neurons[:-1], n_neurons[1:])])

    def feed_forward(self, x):
        """
        Performs a feed forward, storing activation and z
        """
        self.z = []
        self.a = [x]
        self.probabilities = []
        a_temp = x

        for b, w in zip(self.biases, self.weights):
            _z = np.matmul(a_temp, w) + b
            self.z.append(_z)
            a_temp = sigmoid(_z)   # Updating
            self.a.append(a_temp)

        # Softmax
        exp_term = np.exp(z)    # Temporary to reduce FLOPS
        self.probabilities = exp_term / \
            np.sum(exp_term, axis=1, keepdims=True)

    def sgd(self, train_data):
        "stochastic gradient descent"
        for e in range(self.epochs):
            np.random.shuffle(train_data)
            mini_batches = [[train_data[i:i + self.batch_size]
                             for i in range(n, self.batch_size)]]

    def backpropagation(self):
        """ backprop"""
        error_output = [p - self.y_data for p in self.probabilities]
        error_hidden = [np.matmul(error_output, w.T) * a * (1 - a)
                        for w, a in zip(self.weights, self.a)]

        self.weights_gradient = [np.zeros(w.shape) for w in self.weights]
        self.biases_gradient = [np.zeros(b.shape) for b in self.biases]

        # for i in len(error_output):
        #     self.weights_gradient.append()
        #         np.matmul(self.activation[i].T, error_output[i]))
        #     self.biases_gradient.append(np.sum(error_output[i], axis=0))
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


def sigmoid(z):
    """ sigmoid =)))"""
    return 1 / (1 + np.exp(-z))
