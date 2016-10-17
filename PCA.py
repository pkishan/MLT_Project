import scipy.io
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



data = scipy.io.loadmat('mnist_big.mat')

Xtest = data['X_test']
Ytest = np.squeeze(data['Y_test'])

Xtrain = data['X_train']
Ytrain = np.squeeze(data['Y_train'])

dim = [ 50, 100, 200, 300, 500]

for i in dim:
	pca = PCA(n_components=i)
	X_train = Xtrain
	X_test = Xtest 
	pca.fit(X)
	pca.transform(temp1)

	