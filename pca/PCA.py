import scipy.io
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data = scipy.io.loadmat('mnist_big.mat')

Xtrain = data['X_train']
Xtest = data['X_test']
Ytrain = np.squeeze(data['Y_train'])

dim = [50, 75, 100]

for i in dim:
	print i
	pca = PCA(n_components=i)
	# X_train = Xtrain 
	pca.fit(Xtrain)
	X_train = pca.transform(Xtrain)
	X_test = pca.transform(Xtest)
	
	X_train.dump(str(i) + "_pca_train.p")
	X_test.dump(str(i) + "_pca_test.p")
