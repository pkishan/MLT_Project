import scipy.io
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data = scipy.io.loadmat('mnist_big.mat')

Xtrain = data['X_train']
Ytrain = np.squeeze(data['Y_train'])

dim = [50, 75, 100]

for i in dim:
	print i
	pca = PCA(n_components=i)
	X_train = Xtrain 
	pca.fit(Xtrain)
	pca.transform(X_train)
	X_train.dump(str(i) + "_pca.p")
