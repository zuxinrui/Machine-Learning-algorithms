import numpy as np
from numpy import linalg as la
from dataloader import load_data
import matplotlib.pyplot as plt

def eigs(e_vals, e_vecs):
    idx = e_vals.argsort()[::-1]
    eigenValues = e_vals[idx]
    eigenVectors = e_vecs[:, idx]
    return eigenValues, eigenVectors

def pca(i):
    x_train, y_train, x_test, y_test = load_data()
    mean = np.mean(x_train, axis=0)
    meanc = x_train - mean
    cov = np.cov(meanc.T)
    e_vals, e_vecs = la.eig(cov)
    e_vals, e_vecs = eigs(e_vals, e_vecs)
    e_vals = e_vals[0:i]
    e_vecs = e_vecs[:, 0:i]
    return e_vals, e_vecs

def energy(i):
    sume = np.sum(i)
    ener = 0
    for a in range(784):
        ener = ener + i[a]
        if ener/sume>=0.95:
            break
    return a

if __name__ == '__main__':
    a, b = pca(784)
    a1 = b[:, 0]
    a2 = b[:, 1]
    a3 = b[:, 2]
    a4 = b[:, 3]
    a5 = b[:, 4]
    a6 = b[:, 5]
    num1 = np.zeros([28, 28])
    num2 = np.zeros([28, 28])
    num3 = np.zeros([28, 28])
    num4 = np.zeros([28, 28])
    num5 = np.zeros([28, 28])
    num6 = np.zeros([28, 28])
    for i in range(28):
        # print i
        num6[i, :] = a6[28*i:28*(i + 1)]
    en = energy(a)
    print 'over 95% energy dimensions:', en
    # plt.pcolor(num6)
    # plt.colorbar()
    # plt.show()
