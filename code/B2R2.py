# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 10:43:12 2021

@author: eyaraz
"""
import numpy as np
from numpy import diff, cumsum
from numpy.linalg import norm
import glob
from scipy import io, linalg
from math import pi, ceil, floor, exp, pi, log







def high_pass(x, beta, rho, w):
    X_ft_hpf = np.fft.fft(np.fft.ifftshift(x))
    # X_ft_hpf = X_ft_lpf -X_ft_hpf.mean()
    idx_1, idx_2 = np.argmin(abs(w - beta * rho * pi)), np.argmin(abs(w - (2 * pi - beta * rho * pi)))
    X_ft_hpf[:idx_1] = 0
    X_ft_hpf[idx_2 + 1:] = 0
    x_rec_hpf = np.fft.ifftshift(np.fft.ifft(X_ft_hpf))
    return x_rec_hpf


def DFT_matrix(N):
    # i, j = np.meshgrid(np.arange(-int(N/2), int(N/2)), np.arange(-int(N/2), int(N/2)))
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp(- 2 * pi * 1J / N)
    W = np.power(omega, (i - N / 2) * j)
    return W


def PGD(L, x_lambda, delta_rec, change, N_lambda, dev_matrix, reg, step_size, momentum, idx_del):
    ### step_1 - gradient decent ###
    vector = x_lambda + delta_rec
    grad = np.dot(dev_matrix, vector) + reg * delta_rec  # L1:reg * sign(delta_rec)  ,L2:reg * delta_rec
    change = momentum * change + step_size * grad.real
    delta_rec -= change
    ### step_2 - Projection ###
    delta_rec[:int(L / 2) - N_lambda] = 0
    delta_rec[int(L / 2) + N_lambda + 1:] = 0
    if len(idx_del) > 0:
        delta_rec[idx_del] = 0
    return delta_rec, change


def P_Fista(x_lambda, delta_rec, delta_old, N_lambda, dev_matrix, reg, step_size, k, L):
    vector = x_lambda + delta_rec
    grad = np.dot(dev_matrix, vector)
    w_k = k / (k + 3)
    # Accleration
    y = delta_rec + w_k * (delta_rec - delta_old)
    delta_old = delta_rec

    # Prox operator
    prox = y - step_size * grad
    prox[abs(prox) < reg * step_size] = 0
    prox[prox > reg * step_size] = prox[prox > reg * step_size] - reg * step_size
    prox[prox < -reg * step_size] = prox[prox < -reg * step_size] + reg * step_size
    delta_rec = prox
    ### step_2 - Projection ###
    delta_rec[:int(L / 2) - N_lambda] = 0
    delta_rec[int(L / 2) + N_lambda + 1:] = 0
    return delta_rec, delta_old


def quant_delta(delta_rec, Lambda):
    delta_quant_rec = np.round(delta_rec / (2 * Lambda)) * 2 * Lambda
    return delta_quant_rec


class reconstruction_method:

    def __init__(self, x_lambda, X_ft_lambda, M, of, w):
        """
            M - num of wrong samples'
            rho - 2W_nyquist/W_s
            w = [0,2*pi]
        """
        self.x_lambda = x_lambda
        self.X_ft_lambda = X_ft_lambda
        self.M = M
        self.of = of
        self.rho = 1 / of
        self.w = w

    def BBRR(self, Lambda, L):
        x_lambda = self.x_lambda
        rho = self.rho
        M = self.M
        w = self.w
        of = self.of
        N = int(M / 2)

        # Params
        beta = 1.05
        decay = 0.999
        reg = 1e-6
        momentum = 0.9
        epsilon = 1e-3 * (Lambda ** 1.5 * of)

        F = DFT_matrix(L)
        idx_1, idx_2 = np.argmin(abs(w - beta * rho * pi)), np.argmin(abs(w - (2 * pi - beta * rho * pi)))
        diagonal = np.ones((L,))
        diagonal[:idx_1 + 1] = 0
        diagonal[idx_2:] = 0
        D = np.diag(diagonal[:])
        BB = np.dot(D, F)
        dev_matrix = np.dot(BB.conj().T, BB)

        if of == 2:
            step_size = 2 / L
        else:
            step_size = 2 / (idx_2 - idx_1)

        # intialization
        mu = step_size
        mom = momentum
        change = 0
        d1, d2 = 0, 0

        delta_rec = high_pass(-x_lambda.copy(), beta, rho, w)
        delta_rec[:int(L / 2) - N] = 0
        delta_rec[int(L / 2) + N + 1:] = 0
        delta_rec = np.array(delta_rec, dtype=np.float64)

        for iteration in range(1, 1000000):
            # PGD
            delta_rec, change = PGD(L, x_lambda, delta_rec, change, N, dev_matrix, reg, mu, mom, [])
            mu = decay * mu
            mom = decay * mom

            if (iteration % 5) == 0:
                d1_new, d2_new = delta_rec[int(L / 2) - N], delta_rec[int(L / 2) + N]
                if max(abs(d1_new - d1), abs(d2_new - d2)) < epsilon:
                    # print(iteration)
                    delta_rec = quant_delta(delta_rec, Lambda)
                    if N < 1:
                        x_lambda[int(L / 2)] += delta_rec[int(L / 2)]
                        break
                    else:
                        mu = step_size
                        mom = momentum
                        x_lambda[int(L / 2) - N] += delta_rec[int(L / 2) - N]
                        x_lambda[int(L / 2) + N] += delta_rec[int(L / 2) + N]
                        N -= 1
                    d1, d2 = delta_rec[int(L / 2) - N], delta_rec[int(L / 2) + N]
                else:
                    d1, d2 = d1_new, d2_new

        return x_lambda





