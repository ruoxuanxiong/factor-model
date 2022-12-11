import numpy as np
import math


def generate_factor_model(N, T, r=1, mu=0, sigma=1, multiplier=1):
    # generate the factors, loadings and common components, and the state variable
    # r is the number of factors

    # state variable
    S = np.zeros((T,))
    for t in range(T):
        S[t] = 0.1 * multiplier * math.sin(math.pi * math.e * t / T)

    # factor
    F = np.random.normal(mu, sigma, (T, r))
    Lam1 = np.random.normal(mu, sigma, (N, r))
    # perturb loadings by a vector of [1, 1, .., 1, -1, -1, ..., -1] + noise
    Lam2 = np.ones((N, r)) + np.random.normal(mu, sigma / 8, (N, r))
    Lam2[int(N / 2),] = -Lam2[int(N / 2),]
    Cbar = np.zeros((N, T))
    for t in range(T):
        # loading
        Lam = (Lam1 + S[t] * (Lam2 - Lam1))

        # common component
        Cbar[:, t] = Lam.dot(F[t, :])

    return Cbar, S, F, Lam1, Lam2


def generate_error(N, T, mu=0, sigma=1, error_type="iid"):
    # generate error
    e = np.random.normal(mu, sigma, (N, T))

    if error_type == 'heter':
        u = np.random.uniform(low=0.5, high=1.5, size=(N,))
        e = np.diag(u).dot(e)
    elif error_type == 'cross':
        Sigma_e = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                Sigma_e[i, j] = 0.2 ** np.abs(i - j)
        Sigma_e_L = np.linalg.cholesky(Sigma_e)
        e = Sigma_e_L.dot(e)

    return e


def generate_simulation_data(N, T, r=1, mu=0, sigma=1, error_type="iid", multiplier=1):
    # generate the simulated data
    # X is the sum of common component and noise
    Cbar, S, F, Lam1, Lam2 = generate_factor_model(N, T, r, mu, sigma, multiplier=multiplier)

    e = generate_error(N, T, mu, sigma, error_type)

    X = Cbar + e

    return X, Cbar, S, F, Lam1, Lam2