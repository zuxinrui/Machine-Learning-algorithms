import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from dataloader import load_data
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle
import time

x_train, y_train, x_test, y_test = load_data()
start_time = time.time()
pca = PCA(n_components=40)
pca.fit(x_train)
X_train_pca = pca.transform(x_train)
X_test_pca = pca.transform(x_test)
end_time = time.time()
print 'time for PCA:', end_time - start_time
print X_train_pca.shape
print X_test_pca.shape

start_time = time.time()
clf = svm.SVC(kernel="rbf", C=2.8, gamma=.0073)
clf.fit(X_train_pca, y_train)
print 'running...'
y_pred = clf.predict(X_test_pca)
conf_matrix = confusion_matrix(y_test, y_pred)
print conf_matrix
plt.pcolor(conf_matrix)
plt.colorbar()
accu = accuracy_score(y_test, y_pred)
print 'accuracy rate:', accu
end_time = time.time()
print 'time for knn:', end_time - start_time
plt.show()