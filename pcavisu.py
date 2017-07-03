from sklearn import decomposition
import matplotlib.pyplot as plt
from dataloader import load_data
import numpy as np
import seaborn
from mpl_toolkits.mplot3d import Axes3D


x_train, y_train, x_test, y_test = load_data()
X = x_train
y = y_train
pca = decomposition.PCA(n_components=3)
new_X = pca.fit_transform(X)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(new_X[:, 0], new_X[:, 1], new_X[:, 2], c=y, cmap=plt.cm.spectral)
plt.show()