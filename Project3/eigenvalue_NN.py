"""
Neural network for finding eigenvalues of random matrix A using Tensorflow.
A is a random normal distributed NxN matrix. The initial guess of the
eigenvector is a random vector.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})
tf.keras.backend.set_floatx('float64')  # Set default float type
np.random.seed(42)
tf.random.set_seed(42)

N = 6
Nt = 100
scale_factor = N // 6   # To scale the amount neurons in the network

Q = np.random.normal(0, 1, (N, N))
Q = 0.5 * (Q.T + Q)

x = np.random.rand(N)

start = tf.constant(0, dtype=tf.float64)
stop = tf.constant(5, dtype=tf.float64)

_t = tf.linspace(start, stop, Nt + 1)
_t = tf.reshape(_t, (-1, 1))

x0 = tf.convert_to_tensor(x, dtype=tf.float64)
A = tf.convert_to_tensor(Q, dtype=tf.float64)


class DNModel(tf.keras.Model):
    def __init__(self):
        super(DNModel, self).__init__()

        self.dense_1 = tf.keras.layers.Dense(
            48 * scale_factor, activation=tf.keras.activations.sigmoid)
        self.dense_2 = tf.keras.layers.Dense(
            24 * scale_factor, activation=tf.keras.activations.sigmoid)
        self.dense_3 = tf.keras.layers.Dense(
            12 * scale_factor, activation=tf.keras.activations.sigmoid)
        self.out = tf.keras.layers.Dense(N, name='output')

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x = self.dense_3(x)

        return self.out(x)


@tf.function
def rhs(model, x0, t):
    """
    Right hand side of equation. Takes object model from NN, x0: initial
    guess of eigenvector and vector t being time interval.
    """
    g = g_t(model, x0, t)

    F = tf.einsum('ij,ij,kl,il->ik', g, g, A, g) - \
        tf.einsum('ij,jk,ik,il->il', g, A, g, g)

    return F


@tf.function
def g_t(model, x0, t):
    """
    Trial function. Takes object model from NN, x0: initial
    guess of eigenvector and vector t being time interval.
    """
    return tf.einsum('i...,j->ij', tf.exp(-t), x0) + \
        tf.einsum('i...,ij->ij', (1 - tf.exp(-t)), model(t))


# Loss function
@tf.function
def loss(model, x0, t):
    """
    Loss function using mean squared error. Takes object model from NN, x0:
    initial guess of eigenvector and vector t being time interval.
    """
    with tf.GradientTape() as tape:
        tape.watch(t)
        trial = g_t(model, x0, t)

    dt_trial = tape.batch_jacobian(trial, t)
    dt_trial = dt_trial[:, :, 0]

    return tf.losses.MSE(tf.zeros_like(dt_trial), dt_trial - rhs(model, x0, t))


@tf.function
def grad(model, x0, t):
    """
    Gradient. Takes object model from NN, x0: initial guess of eigenvector and
    vector t being time interval.
    """
    with tf.GradientTape() as tape:
        loss_value = loss(model, x0, t)

    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def NN_solver():
    """
    Finds the eigenvector using a neural network. Also calculates the eigenvalue
    corresponding to eigenvector for last model. Returns eigenvalues and
    eigenvector for all time points.
    """
    epochs = 5000 * scale_factor

    model = DNModel()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    # Training
    for epoch in range(epochs):
        cost, gradients = grad(model, x0, _t)
        optimizer.apply_gradients(
            zip(gradients, model.trainable_variables))

        print('  \r', end='')
        print(f'Step: {optimizer.iterations.numpy()}, '
              + f'Loss: {tf.reduce_mean(cost.numpy())}', flush=True, end='')

        eigvecs = g_t(model, x0, _t)
        temp = eigvecs / tf.sqrt(tf.einsum('ij,ij->i',
                                           eigvecs, eigvecs)[:, tf.newaxis])
        eigvals = tf.einsum('ij,ij->i', tf.matmul(temp, A), temp)

    return eigvals, eigvecs


def main():
    """
    Calls the neural network and forward euler to calculate eigenvalues and
    eigenvector. Shows comparison plots of eigenvector and eigenvalue with
    values found by numpy linalg, NN and FE.
    """
    eigvals, eigvecs = NN_solver()
    eigvals_FE, eigvecs_FE, iters = FE_eigen()

    # Eigenvector and values from numpy linalg
    w, v = np.linalg.eig(Q)
    idxmax = np.argmax(w)   # Index of largest eigenvalue

    # Normalizing time evolution of eigenvectors based on the final eigenvector
    k = np.sqrt(eigvecs[-1, :].numpy().T @ eigvecs[-1, :].numpy())
    if np.sign(eigvecs[-1, 0]) != np.sign(v[0, idxmax]):
        k *= -1  # If eigenvector is shiften sign
    eigvecs /= k

    # Linspace
    t = np.linspace(0, 5, eigvals.shape[0])
    t_FE = np.linspace(0, 5, iters)

    # Plotting time evolution eigenvalue
    fig = plt.figure()
    plt.plot(t, w[idxmax] * np.ones(len(t)), '--', label='$\\lambda_{max}$')
    plt.plot(t_FE, eigvals_FE, label='FE')
    plt.plot(t, eigvals, label='NN')
    plt.xlabel('Time $t$')
    plt.ylabel('Eigenvalue')
    plt.legend()
    plt.show()

    # Plotting time evolution of eigenvector elements
    plot_colour = ['r', 'c', 'b', 'g', 'm', 'k']
    fig = plt.figure()
    for i, colour in enumerate(plot_colour):
        plt.plot(t, v[i, idxmax] * np.ones(len(t)), '--' + colour)
        plt.plot(t, eigvecs[:, i], colour)
    plt.xlabel('Time $t$')
    plt.ylabel('Elements of eigenvector')
    plt.legend([f'NN $\\lambda = {eigvals[-1]:.6f}$',
                # f'FDM $\\lambda = {eigvals_FE[-1]:.6f}$',
                f'Exact $\\lambda = {w[idxmax]:.6f}$',
                '- NN', '-- Exact'], handletextpad=0, handlelength=0,
               loc='lower center')
    plt.show()


def FE_eigen():
    """
    Eigenvalue solver using forward euler. Returns array of eigenvalues,
    eigenvector and integer showing maximum iterations (for plotting).
    """
    x0 = x
    A = Q

    dt = 0.001
    max_iters = 20000

    eigvals = []
    eigvec = []

    for i in range(max_iters):
        x0 = x0 + dt * (np.matmul(np.matmul(x0.T, x0) * A, x0)
                        - np.matmul(np.matmul(x0.T, A), x0) * x0)

        eigvals.append(np.matmul(np.matmul(x0.T, A), x0) / np.matmul(x0.T, x0))
        eigvec.append(x0)

    return np.array(eigvals), np.array(eigvec), max_iters


if __name__ == '__main__':
    main()
