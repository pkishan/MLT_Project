import scipy.io
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


Xtrain = np.load("small_X_train.p")
Y_train = np.load("small_Y_train.p")

data = scipy.io.loadmat('mnist_big.mat')

Xtest = data['X_test']
Y_test = np.squeeze(data['Y_test'])

# Xtrain = data['X_train']
# Y_train = np.squeeze(data['Y_train'])

dim = [ 50, 100, 200, 300, 500]

for i in dim:
	pca = PCA(n_components=i) 
	pca.fit(Xtrain)
	X_train = pca.transform(Xtrain)
	X_test = pca.transform(Xtest)

	print X_train.shape

