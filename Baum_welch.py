import pandas as pd
import numpy as np

def forward(V, a, b, pi):
    alpha = np.zeros((V.shape[0], a.shape[0]))
    alpha[0, :] = pi * b[:, V[0]]
    for t in range(1, V.shape[0]):
        for j in range(a.shape[0]):
            alpha[t, j] = alpha[t - 1].dot(a[:, j]) * b[j, V[t]]
    return alpha

def backward(V, a, b):
    beta = np.zeros((V.shape[0], a.shape[0]))
    beta[V.shape[0] - 1] = np.ones((a.shape[0]))
    for t in range(V.shape[0] - 2, -1, -1):
        for j in range(a.shape[0]):
            beta[t, j] = (beta[t + 1] * b[:, V[t + 1]]).dot(a[j, :])
    return beta

def baum_welch(V, a, b, pi, n_iter=100):
    M = a.shape[0]
    T = len(V)
    for n in range(n_iter):
        alpha = forward(V, a, b, pi)
        beta = backward(V, a, b)
        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            denominator = np.dot(np.dot(alpha[t, :].T, a) * b[:, V[t + 1]].T, beta[t + 1, :])
            for i in range(M):
                numerator = alpha[t, i] * a[i, :] * b[:, V[t + 1]].T * beta[t + 1, :].T
                xi[i, :, t] = numerator / denominator
        gamma = np.sum(xi, axis=1)
        a = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))
        gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))
        K = b.shape[1]
        denominator = np.sum(gamma, axis=1)
        for l in range(K):
            b[:, l] = np.sum(gamma[:, V == l], axis=1)
        b = np.divide(b, denominator.reshape((-1, 1)))
    return {"a":a, "b":b}

data = pd.read_csv('data_python.csv')
V = data['Visible'].values
a = np.ones((2, 2))
a = a / np.sum(a, axis=1)
b = np.array(((1, 3, 5), (2, 4, 6)))
b = b / np.sum(b, axis=1).reshape((-1,1))
pi = np.array((0.5, 0.5))
print(baum_welch(V, a, b, pi, n_iter=100))