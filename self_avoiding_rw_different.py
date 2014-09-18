#! /usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto, June 2014.

import numpy as np


def self_avoiding_rw_Rosenbluth(N, x0=0, y0=0):

    walker = 100
    W = np.zeros([walker, N], 'f')
    R_2 = np.zeros([walker, N], 'i')

    for m in xrange(walker):
        x, y = x0 + N + 1, y0 + N + 1
        lattice = np.zeros([2 * N + 3, 2 * N + 3], dtype=bool)
        lattice[x][y] = True
        lattice[x][y + 1] = True
        x, y = x, y + 1
        W[m][0] = 1
        R_2[m][0] = 1
        for n in xrange(1, N):
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

    ave_R_2 = np.sum(W * R_2, axis=0, dtype=np.float32) \
        /                                       \
        np.sum(W, axis=0, dtype=np.float32)

    print W * R_2
    print ave_R_2

if __name__ == '__main__':

    N = 4
    self_avoiding_rw_Rosenbluth(N)
