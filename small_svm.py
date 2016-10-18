import numpy as np
import scipy.io
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = scipy.io.loadmat('mnist_big.mat')

X_train = np.load("small_X_train.p")
Y_train = np.load("small_Y_train.p")

X_test = data['X_test'][1:2000,:]
Y_test = np.squeeze(data['Y_test'])[1:2000]

# default parameters rbf-kernel
clf = SVC() 

print "creating model ..."
clf.fit(X_train, Y_train)
print "model created"

print "predicting test data ..."
prediction = clf.predict(X_test)

accuracy = accuracy_score(Y_test, prediction)

print accuracy