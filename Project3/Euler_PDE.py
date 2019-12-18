"""
Solves a partial differential equation using finite differencing with forward
euler, compares with the analytical solution.
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})


def analytical(x, t, L=1):
    """
    Analytical solution. x: array over spatial domain 0->L, t: array over time
    domain.
    """
    return np.exp(-np.pi * np.pi * t) * np.sin(np.pi * x)


def solver(dx,
           beta=0.9,    # Stability criterion, valid for beta = [0,1]
           plot=False
           ):
    """
    Solves a PDE by finite differencing using forward euler. Returns an array u
    of the solutions of the PDE, vector of spatial and time domain x and t,
    respectively and the number of points in space and time domain nx and nt.
    """
    beta = 0.9  # Stability criterion
    dt = beta * dx**2 / 2

    L = 1
    T = 1

    nt = int(T / dt)
    nx = int(L / dx)

    x = np.linspace(0, L, nx + 1)
    t = np.linspace(0, T, nt + 1)

    u = np.zeros((t.shape[0], x.shape[0]))
    # Initial condition, preserves boundary conditions (sin(0) = sin(pi*L/L) = 0)
    u[0, :] = np.sin(np.pi * x)

    if plot:
        fig = plt.figure(111)
        ax = fig.add_subplot(111)

        plt.plot(analytical(x, t[0]), label='Analytical')
        plt.plot(u[0, :], label='Numerical')
        plt.title(f't={t[0]:.3f}, n={0}')

    for n in range(t.shape[0] - 1):
        u[n + 1, 1:-1] = u[n, 1:-1] + dt / dx**2 * \
            (u[n, 2:] - 2 * u[n, 1:-1] + u[n, :-2])

        # Boundaries
        u[n + 1, 0] = 0
        u[n + 1, -1] = 0

        if plot:
            ax.clear()
            ax.set_ylim(-1.5, 1.5)
            ax.plot(x, u[n + 1, :])
            ax.plot(x, analytical(x, t[n]))
            ax.set_title(f't={t[n]:.2f}, n={n}')
            plt.legend(['Numerical', 'Analytical'])
            plt.pause(0.0001)

    return u, x, t, nx, nt


def main(dx=0.1):
    """
    Plots the numerical against the analytical solution for two time steps t_1
    and t_2 and prints maximum error, relative error and maximum absolute error.
    """
    u, x, t, _, _ = solver(dx)

    # Choosing t1 and t2
    idx1 = int(0.025 * t.shape[0])
    idx2 = int(0.45 * t.shape[0])

    u_t1 = u[idx1, :]
    t1 = t[idx1]
    u_analyt_t1 = analytical(x, t1)

    u_t2 = u[idx2, :]
    t2 = t[idx2]
    u_analyt_t2 = analytical(x, t2)

    diff = u_t1 - u_analyt_t1
    diff2 = u_t2 - u_analyt_t2
    max_err = np.max(np.abs(diff[1:-1]))
    max_err2 = np.max(np.abs(diff2[1:-1]))

    rel_err = np.nanmax(np.abs(diff[1:-1] / u_analyt_t1[1:-1]))
    rel_err2 = np.nanmax(np.abs(diff2[1:-1] / u_analyt_t2[1:-1]))

    plt.plot(np.abs(diff), label='abs($u_{t_1}$ - analytical)')
    plt.xlabel('$x$')
    plt.ylabel('Difference')
    plt.legend()
    plt.show()

    plot(x, u_t1, u_analyt_t1, 'numerical $t_1$', '$x$', '$u(x,t_1)$')
    plot(x, u_t2, u_analyt_t2, 'numerical $t_2$', '$x$', '$u(x,t_2)$')

    print(f'Max error for t1: {max_err}, max error for t2: {max_err2}')
    print(f'Relative error for t1: {rel_err}, relative error for t2: {rel_err2}')
    print(f'Absolute error for t1: {np.max(np.abs(diff[1:-1]))}, absolute error for t2: {np.max(np.abs(diff2[1:-1]))}')


def plot(x, y1, y2, label, xlab, ylab):
    plt.plot(x, y, label=label)
    plt.plot(x, y2, '--', label=label)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main(dx=0.01)
