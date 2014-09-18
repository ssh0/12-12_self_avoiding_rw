#! /usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto, June 2014.

import numpy as np


class SAW:

    """ Class for the simulation of SAW (Self-Avoiding Random Walk).

    functions:
    self_avoiding_rw_d2(N, x0=0, y0=0)
    self_avoiding_rw_Rosenbluth(N, x0=0, y0=0)
    """

    def __init__(self, walker=1000, x0=0, y0=0):
        self.walker = walker
        self.x0 = x0
        self.y0 = y0

    def self_avoiding_rw_d2(self, N, x0=0, y0=0):

        walker = self.walker
        f = np.array([[1, 0], [0, 1]])
        r = np.array([[0, 1], [-1, 0]])
        l = np.array([[0, -1], [1, 0]])
        R_2 = np.zeros([walker, N], 'i')

        for m in range(walker):
            x, y = x0 + N + 1, y0 + N + 1
            lattice = np.zeros([2 * N + 3, 2 * N + 3], dtype=bool)
            lattice[x][y] = True
            lattice[x][y + 1] = True
            _x, _y = x, y
            x, y = x, y + 1
            R_2[m][0] = 1
            for n in range(1, N):

                vec = x - _x, y - _y
                _x, _y = x, y

                p = np.random.rand() * 3
                if p < 1:     # direction = 'forward'
                    x, y = np.dot(f, vec) + (x, y)
                elif p < 2:   # direction = 'right'
                    x, y = np.dot(r, vec) + (x, y)
                else:         # direction = 'left'
                    x, y = np.dot(l, vec) + (x, y)

                if lattice[x][y]:
                    break
                else:
                    lattice[x][y] = True
                    R_2[m][n] = (x - x0 - N - 1) ** 2 + (y - y0 - N - 1) ** 2

        active_walkers = np.zeros(N)
        for m in range(N):
            i = 0
            for k in range(walker):
                if R_2[k][m]:
                    i += 1
            active_walkers[m] = i

        def f(N):
            return active_walkers[N - 1] / float(walker)

        def ave_R_2():
            return np.sum(R_2, axis=0, dtype=np.float32) / active_walkers

        self.f_N = [f(n) for n in range(1, N + 1)]
        self.ave_R_2 = ave_R_2()

    def self_avoiding_rw_Rosenbluth(self, N, x0=0, y0=0):

        walker = self.walker
        W = np.zeros([walker, N], 'f')
        R_2 = np.zeros([walker, N], 'i')

        for m in range(walker):
            x, y = x0 + N + 1, y0 + N + 1
            lattice = np.zeros([2 * N + 3, 2 * N + 3], dtype=bool)
            lattice[x][y] = True
            lattice[x][y + 1] = True
            x, y = x, y + 1
            W[m][0] = 1
            R_2[m][0] = 1
            for n in range(1, N):
                path = []
                if not lattice[x - 1][y]:
                    path.append((x - 1, y))
                if not lattice[x + 1][y]:
                    path.append((x + 1, y))
                if not lattice[x][y - 1]:
                    path.append((x, y - 1))
                if not lattice[x][y + 1]:
                    path.append((x, y + 1))

                if len(path) == 0:
                    break
                if len(path) == 1:
                    x = path[0][0]
                    y = path[0][1]
                    W[m][n] = W[m][n - 1] / 3.
                if len(path) == 2:
                    p = np.random.rand()
                    if p < 0.5:
                        x = path[0][0]
                        y = path[0][1]
                    else:
                        x = path[1][0]
                        y = path[1][1]
                    W[m][n] = (W[m][n - 1] * 2.) / 3.
                if len(path) == 3:
                    p = np.random.rand() * 3
                    if p < 1:
                        x = path[0][0]
                        y = path[0][1]
                    elif p < 2:
                        x = path[1][0]
                        y = path[1][1]
                    else:
                        x = path[2][0]
                        y = path[2][1]
                    W[m][n] = W[m][n - 1]

                lattice[x][y] = True
                R_2[m][n] = (x - x0 - N - 1) ** 2 + (y - y0 - N - 1) ** 2

        self.ave_R_2 = np.sum(W * R_2, axis=0, dtype=np.float32) \
            /                                       \
            np.sum(W, axis=0, dtype=np.float32)


def plot_graph(x_data, y_data, x_labels, y_labels,
               xscale='linear', yscale='linear', aspect='auto'):
    """ Plot the graph about y_data for each x_data.
    """
    import matplotlib.pyplot as plt

    d = len(y_data)
    if not len(x_data) == len(y_data) == len(x_labels) == len(y_labels):
        raise ValueError("Arguments must have the same dimension.")
    if d == 0:
        raise ValueError("At least one data for plot.")
    if d > 9:
        raise ValueError("""So much data for plot in one figure.
                            Please divide two or more data sets.""")

    fig = plt.figure(figsize=(9, 8))
    subplot_positioning = [
        '11', '21', '22', '22', '32', '32', '33', '33', '33']
    axes = []
    for n in range(d):
        lmn = int(subplot_positioning[d - 1] + str(n + 1))
        axes.append(fig.add_subplot(lmn))

    for i, ax in enumerate(axes):
        ymin, ymax = min(y_data[i]), max(y_data[i])
        ax.set_aspect(aspect)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlabel(x_labels[i], fontsize=16)
        ax.set_ylabel(y_labels[i], fontsize=16)
        ax.set_ymargin(0.05)
        ax.plot(x_data[i], y_data[i])

    fig.subplots_adjust(wspace=0.2, hspace=0.5)
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':

    def ex_a(N):
        rw = SAW()
        rw.self_avoiding_rw_d2(N)

        print 'f(%d) = ' % N + str(rw.f_N[N - 1])
        print '<R^{2}(%d)> = ' % N + str(rw.ave_R_2[N - 1])

        x_labels = [r'$N$'] * 2
        y_labels = [r'$f(N)$', r'$<R^{2}(N)>$']
        plot_graph([range(1, N + 1)] * 2, [
                   rw.f_N, rw.ave_R_2], x_labels, y_labels)

    def ex_b(N):
        rw = SAW()
        rw.self_avoiding_rw_Rosenbluth(N)

        for n in [4, 8, 16, 32]:
            print '<R^{2}(%d)> = ' % n + str(rw.ave_R_2[n - 1])

        x_labels = [r'$N$']
        y_labels = [r'$<R^{2}(N)>$']
        plot_graph([range(1, N + 1)], [rw.ave_R_2], x_labels, y_labels,
                   xscale='log', yscale='log', aspect='equal')

    def ex_b_fit(N):
        import scipy.optimize as optimize
        from math import sqrt

        trial = 100
        nu = np.zeros(trial)
        for i in range(trial):
            rw = SAW()
            rw.self_avoiding_rw_Rosenbluth(N)

            parameter0 = [1.0, 0.75]  # C, nu

            def fit_func(parameter0, n, r_2):
                C = parameter0[0]
                nu = parameter0[1]
                residual = r_2 - C * (n ** (2 * nu))
                return residual

            result = optimize.leastsq(fit_func, parameter0,
                                      args=(np.array(range(1, N + 1)), rw.ave_R_2))

            nu[i] = result[0][1]

        print 'nu =', np.average(nu), u'pm', np.std(nu) / sqrt(rw.walker)

#    ex_a(N=25)
#    ex_b(N=32)
    ex_b_fit(N=32)
