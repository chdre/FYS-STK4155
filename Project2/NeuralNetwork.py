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
        self.b = np.asarray([np.zeros(y) + 0.01 for y in self.sizes[1:]])
        self.w = np.asarray([np.random.randn(x, y) for x, y in zip(
            self.sizes[:-1], self.sizes[1:])])

    def set_activation_function(self, i):
        if isinstance(self.activation_func, (list, tuple, np.ndarray)):
            self.act_func = self.activation_func[i]
        elif isinstance(self.activation_func, str):
            self.act_func = self.activation_func

    def feed_forward(self):
        """
        Performs a feed forward, storing activation and z
        """
        # self.z = np.empty(len(self.b) + 1, dtype=np.ndarray)
        self.a = np.empty(self.layers, dtype=np.ndarray)
        self.a[0] = self.x_data

        # self.forward_activation(self.x_data) "sett activation på ny"
        # self.z[0] = 0

        for l in range(self.layers - 1):
            if l < self.layers - 1:
                self.set_activation_function(l)
            # print(self.w[l].shape, self.a[l].shape, self.b[l].shape)
            # self.z[l] = np.matmul(self.a[l - 1], self.w[l]) + self.b[l]
            # self.z[l + 1] =
            z = (self.a[l] @ self.w[l]) + self.b[l]
            self.a[l + 1] = self.forward_activation(z)  # (self.z[l + 1])

    def backpropagation(self):
        """ backprop"""
        delta = np.empty(self.layers - 1, dtype=np.ndarray)
        self.w_grad = np.empty(self.layers - 1, dtype=np.ndarray)
        self.b_grad = np.empty(self.layers - 1, dtype=np.ndarray)

        # Delta for output layer
        delta[-1] = self.a[-1] - self.y_data

        # Gradients for output layer
        self.w_grad[-1] = self.a[-2].T @ delta[-1]
        self.b_grad[-1] = np.sum(delta[-1], axis=0)

        for l in range(0, self.layers - 2):
            # Calculating gradient for hidden layer(s)
            self.set_activation_function(l)
            # print(f"layer: {l}, a = {self.act_func}")

            delta[l] = delta[l + 1] @ self.w[l + 1].T + self.cost_derivative(self.a[l + 1])
            self.w_grad[l] = self.a[l].T @ delta[l]
            self.b_grad[l] = np.sum(delta[l], axis=0)

        for i in range(self.layers - 1):
            self.w[i] -= self.eta * self.w_grad[i]
            self.b[i] -= self.eta * self.b_grad[i]

        # for l in range(1, self.layers - 1):
        #     self.set_activation_function(-l)
        #
        #     delta[-l - 1] = (delta[-l] @ self.w[-l].T) * self.cost_derivative(self.a[-l])
        #
        #     self.w_grad[-l] = self.a[-l].T @ delta[-l - 1]
        #     self.b_grad[-l] = np.sum(error[-l], axis=0)
        #
        #     if self.lmbda > 0:
        #         self.w_grad[-l] += self.lmbda * self.w[-l]
        #
        #     self.w[-l] -= self.eta * self.w_grad[-l]
        #     self.b[-l] -= self.eta * self.b_grad[-l]

    def forward_activation(self, z):
        if self.act_func == 'sigmoid':
            # print("sig")
            return 1 / (1 + np.exp(-z))
        elif self.act_func == 'tanh':
            return np.tanh(z)
        elif self.act_func == 'relu':
            return np.maximum(0, z)
        elif self.act_func == 'leaky_relu':
            return np.maximum(self.leaky_slope * z, z)
        elif self.act_func == 'softmax':
            # print("soft")
            exp_term = np.exp(z)
            return exp_term / np.sum(exp_term, axis=1, keepdims=True)

    def cost_derivative(self, a):
        if self.act_func == 'sigmoid':
            return a * (1 - a)
        elif self.act_func == 'tanh':
            return 1 - np.square(a)
        elif self.act_func == 'relu':
            return 1 * (a > 0)
        elif self.act_func == 'leaky_relu':
            d = np.zeros_like(a)
            d[a <= 0] = self.leaky_slope
            d[a > 0] = 1
            return d
        elif self.act_func == 'softmax':
            return a

    def predict(self, x):
        z = np.empty(len(self.b), dtype=np.ndarray)
        a = np.empty(len(self.b) + 1, dtype=np.ndarray)
        a[0] = x

        for i in range(len(self.b)):
            self.set_activation_function(i)
            _z = np.matmul(a[i], self.w[i]) + self.b[i]
            z[i] = _z
            a[i + 1] = self.forward_activation(_z)

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
