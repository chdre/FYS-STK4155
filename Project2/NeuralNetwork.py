import numpy as np

np.random.seed(42)


class NeuralNetwork:
    def __init__(
            self,
            x_data,
            y_data,
            sizes,  # list of shape_x, nodes, shape_y
            activation_function=['sigmoid', 'softmax'],
            leaky_slope=0.1,
            cost_function='notregression',
            eta=0.1,
            lmbda=0,
            epochs=10,
            batch_size=100):
        self.x_data_full = x_data
        self.y_data_full = y_data

        self.sizes = sizes
        self.n_inputs = x_data.shape[0]
        self.layers = len(sizes)

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // x_data.shape[1]
        self.eta = eta
        self.lmbda = lmbda

        self.activation_func = activation_function
        self.leaky_slope = leaky_slope
        self.cost_func = cost_function

        self.create_bias_and_weights()

    def create_bias_and_weights(self):
        self.b = np.asarray([np.zeros(y) + 0.01 for y in self.sizes[1:]])
        self.w = np.asarray([np.random.randn(x, y) for x, y in zip(
            self.sizes[:-1], self.sizes[1:])])

    def feed_forward(self):
        """
        Performs a feed forward, storing activation and z
        """

        self.a = np.empty(self.layers, dtype=np.ndarray)
        self.a[0] = self.x_data

        for l in range(self.layers - 1):
            z = (self.a[l] @ self.w[l]) + self.b[l]
            self.a[l + 1] = self.forward_activation(z, l)

    def cost_function(self):
        if not self.cost_func == 'regression':
            return self.act_func_derivative(self.a[-1], -1) * (self.a[-1] - self.y_data)
        else:
            return (self.a[-1] - self.y_data)

    def backpropagation(self):
        """ backprop"""
        delta = np.empty(self.layers - 1, dtype=np.ndarray)
        self.w_grad = np.empty(self.layers - 1, dtype=np.ndarray)
        self.b_grad = np.empty(self.layers - 1, dtype=np.ndarray)

        # Delta for output layer
        delta_old = self.cost_function()

        # Gradients for output layer
        self.w_grad[-1] = self.a[-2].T @ delta_old
        self.b_grad[-1] = np.sum(delta_old, axis=0)

        for l in range(self.layers - 2, 0, -1):
            delta = (delta_old @ self.w[l].T) \
                * self.act_func_derivative(self.a[l], l - 1)
            self.w_grad[l - 1] = self.a[l - 1].T @ delta
            self.b_grad[l - 1] = np.sum(delta, axis=0)

            if self.lmbda > 0:
                self.w_grad[l] += self.lmbda * self.w[l]

            delta_old = delta

        for i in range(self.layers - 1):
            self.w[i] -= self.eta * self.w_grad[i]
            self.b[i] -= self.eta * self.b_grad[i]

    def forward_activation(self, z, index):
        self.act_func = self.activation_func[index]
        if self.act_func == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif self.act_func == 'tanh':
            return np.tanh(z)
        elif self.act_func == 'relu':
            return np.maximum(0, z)
        elif self.act_func == 'leaky_relu':
            return np.maximum(self.leaky_slope * z, z)
        elif self.act_func == 'softmax':
            exp_term = np.exp(z)
            return exp_term / np.sum(exp_term, axis=1, keepdims=True)
        elif self.act_func == 'nothing':
            return z

    def act_func_derivative(self, a, index):
        self.act_func = self.activation_func[index]
        if self.act_func == 'sigmoid':
            return a * (1 - a)
        elif self.act_func == 'tanh':
            return 1 - np.square(a)
        elif self.act_func == 'relu':
            return np.heaviside(a, 0)
        elif self.act_func == 'leaky_relu':
            d = np.zeros_like(a)
            d[a <= 0] = self.leaky_slope
            d[a > 0] = 1
            return d

    def predict(self, x):
        # if not self.cost_func == 'regression':
        a = np.empty(self.layers, dtype=np.ndarray)
        # self.a[0] = x
        #
        # self.feed_forward()
        # probability = np.round(self.a[-1])
        #
        # return probability
        #
        # else:
        # a = np.empty(self.layers, dtype=np.ndarray)
        a[0] = x

        for l in range(self.layers - 1):
            z = (a[l] @ self.w[l]) + self.b[l]
            a[l + 1] = self.forward_activation(z, l)

        if self.cost_function == 'regression':
            probability = a[-1]
        else:
            probability = np.round(a[-1])

        return probability

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
