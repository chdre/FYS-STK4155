"""
Solves a partial differential equation with a neural network and compares
to the analytical solution.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import matplotlib.ticker as mtick
from sklearn import metrics

plt.rcParams.update({'font.size': 12})
tf.keras.backend.set_floatx('float64')  # Set default float type
tf.random.set_seed(42)


class DNModel(tf.keras.Model):
    def __init__(self):
        super(DNModel, self).__init__()

        self.dense_1 = tf.keras.layers.Dense(
            50, activation=tf.nn.sigmoid)
        self.dense_2 = tf.keras.layers.Dense(
            25, activation=tf.nn.sigmoid)
        self.out = tf.keras.layers.Dense(1, name='output')

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)

        return self.out(x)


@tf.function
def I(x):
    "Initial condition. x is a vector."
    return tf.sin(np.pi * x)


@tf.function
def trial_solution(model, x, t):
    """
    Trial solution. Takes object model from NN, x vector of position and vector
    t is time interval.
    """
    point = tf.concat([x, t], axis=1)
    return(1 - t) * I(x) + x * (1 - x) * t * model(point)

# Loss function
@tf.function
def loss(model, x, t):
    """
    Loss function. Takes object model from NN, x vector of position and vector
    t is time interval.
    """
    with tf.GradientTape() as tape:
        tape.watch([x, t])
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([x, t])
            trial = trial_solution(model, x, t)

        dx_trial = tape2.gradient(trial, x)
        dt_trial = tape2.gradient(trial, t)
    d2x_trial = tape.gradient(dx_trial, x)

    del tape2  # Deleting tape2, persistent

    return tf.losses.MSE(tf.zeros_like(d2x_trial), d2x_trial - dt_trial)


@tf.function
def grad(model, x, t):
    """
    Gradient. Takes object model from NN, x vector of position and vector
    t is time interval.
    """
    with tf.GradientTape() as tape:
        loss_value = loss(model, x, t)

    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def main(dx):
    """
    Solves a PDE with a neural network. Compares the solution of forward euler
    and the neural network with the analytical solution by plots, prints the
    mean squared, relative and max error for the whole time domain and three
    time steps.
    """
    from euler_PDE import solver, analytical
    u_fdm, x_fdm, t_fdm, Nx, Nt = solver(dx)

    N = 100

    start = tf.constant(0, dtype=tf.float64)
    stop = tf.constant(1, dtype=tf.float64)

    x, t = tf.meshgrid(tf.reshape(tf.linspace(start, stop, N + 1), (-1, 1)),
                       tf.reshape(tf.linspace(start, stop, N + 1), (-1, 1)))
    x, t = tf.reshape(x, (-1, 1)), tf.reshape(t, (-1, 1))

    learning_rate = 0.01
    epochs = 100
    loss_vals = np.zeros(epochs)

    model = DNModel()

    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # Training
    for epoch in range(epochs):
        cost, gradients = grad(model, x, t)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        loss_vals[epoch] = tf.reduce_mean(cost.numpy())

        print('\r', end='')
        print(f'Step: {optimizer.iterations.numpy()}, '
              + f'Loss: {tf.reduce_mean(cost.numpy())}', flush=True, end='')
    print('')

    X, T = tf.meshgrid(tf.reshape(tf.linspace(start, stop, Nx + 1), (-1, 1)),
                       tf.reshape(tf.linspace(start, stop, Nt + 1), (-1, 1)))
    x, t = tf.reshape(X, (-1, 1)), tf.reshape(T, (-1, 1))

    xx = np.linspace(0, 1, Nx + 1)
    tt = np.linspace(0, 1, Nt + 1)

    u_dnn = tf.reshape(trial_solution(model, x, t), (Nt + 1, Nx + 1))
    analyt = np.reshape(analytical(x, t), (Nt + 1, Nx + 1))

    diff_au = analyt - u_dnn
    diff_af = analyt - u_fdm
    print(f'Max abs. error between analytical and DNN: {np.nanmax(np.abs(diff_au))}')
    print(f'Max relative error, analytical and DNN: {np.nanmax(np.abs(diff_au / analyt))}')
    print(f'Mean squared error, analytical and DNN: {metrics.mean_squared_error(analyt,u_dnn)}')
    print(f'Max abs. error between analytical and FDM: {np.nanmax(np.abs(diff_af))}')
    print(f'Max relative error, analytical and FDM: {np.nanmax(np.abs(diff_af / analyt))}')
    print(f'Mean squared error, analytical and FDM: {metrics.mean_squared_error(analyt,u_fdm)}')

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title(f'')
    s = ax.plot_surface(T, X, np.abs(diff_au), linewidth=0,
                        antialiased=False, cmap=cm.viridis)
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('Position $x$')

    idx1 = int(Nt * 0.05)
    idx2 = int((Nt + 1) / 2)
    idx3 = Nt

    t1 = tt[idx1]
    t2 = tt[idx2]
    t3 = tt[idx3]

    u_dnn_1 = u_dnn[idx1, :]
    u_dnn_2 = u_dnn[idx2, :]
    u_dnn_3 = u_dnn[idx3, :]

    analyt_1 = analyt[idx1, :]
    analyt_2 = analyt[idx2, :]
    analyt_3 = analyt[idx3, :]

    u_fdm_1 = u_fdm[idx1, :]
    u_fdm_2 = u_fdm[idx2, :]
    u_fdm_3 = u_fdm[idx3, :]

    print(f'Max relative error, DNN analytical at t1: \
        {np.nanmax(np.abs((analyt_1[1:-1] - u_dnn_1[1:-1])/analyt_1[1:-1]))}')
    print(f'Max relative error, DNN analytical at t2: \
        {np.nanmax(np.abs((analyt_2[1:-1] - u_dnn_2[1:-1])/analyt_2[1:-1]))}')
    print(f'Max relative error, DNN analytical at t3: \
        {np.nanmax(np.abs((analyt_3[1:-1] - u_dnn_3[1:-1])/analyt_3[1:-1]))}')

    print(f'Max relative error, FDM analytical at t1: \
        {np.nanmax(np.abs((analyt_1[1:-1] - u_fdm_1[1:-1])/analyt_1[1:-1]))}')
    print(f'Max relative error, FDM analytical at t2: \
        {np.nanmax(np.abs((analyt_2[1:-1] - u_fdm_2[1:-1])/analyt_2[1:-1]))}')
    print(f'Max relative error, FDM analytical at t3: \
        {np.nanmax(np.abs((analyt_3[1:-1] - u_fdm_3[1:-1])/analyt_3[1:-1]))}')
    print(f'Mean squared error, analytical and DNN: \
        {metrics.mean_squared_error(analyt,u_dnn)}')

    plot_computed_sol(xx, u_dnn_1, analyt_1, u_fdm_1, '$u(x, t_1)$')
    plot_computed_sol(xx, u_dnn_2, analyt_2, u_fdm_2, '$u(x, t_2)$')
    plot_computed_sol(xx, u_dnn_3, analyt_3, u_fdm_3, '$u(x, t_3)$')


def plot_computed_sol(x, y1, y2, y3, ylab):
    "Plot y1, y2 and y3 as a function of x. "
    plt.figure()
    plt.plot(x, y1, label='DNN')
    plt.plot(x, y3, label='FDM')
    plt.plot(x, y2, '--', label='Analytical')
    plt.xlabel('$x$')
    plt.ylabel(ylab)
    plt.tight_layout()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main(0.01)
