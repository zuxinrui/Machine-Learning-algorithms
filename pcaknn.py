import numpy as np
from numpy import arange
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics
from dataloader import load_data
import time

x_train, y_train, x_test, y_test = load_data()
start_time = time.time()
pca = PCA(n_components=153)
pca.fit(x_train)

X_train_pca = pca.transform(x_train)
X_test_pca = pca.transform(x_test)
end_time = time.time()
print 'time for PCA:', end_time - start_time
print X_train_pca.shape
print X_test_pca.shape


n = 5
print 'neighbours:', n
start_time = time.time()
clf = KNeighborsClassifier(n_neighbors=n)
clf.fit(X_train_pca, y_train)
print 'running...'
y_pred = clf.predict(X_test_pca)
conf_matrix = confusion_matrix(y_test, y_pred)
print conf_matrix
plt.pcolor(conf_matrix)
plt.colorbar()
plt.show()
accu = accuracy_score(y_test, y_pred)
print 'accuracy rate:', accu
end_time = time.time()
print 'time for knn:', end_time - start_time