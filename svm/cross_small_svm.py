import numpy as np
import scipy.io

from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

f1=open('./small_svm_cross_val.txt', 'w+')

data = scipy.io.loadmat('../mnist_big.mat')

X_train = np.load("../small_X_train.p")
Y_train = np.load("../small_Y_train.p")

X_test = data['X_test'][1:2000,:]
Y_test = np.squeeze(data['Y_test'])[1:2000]

# default parameters rbf-kernel
clf = SVC() 

print >> f1, "Using cross validation \n"

kfold = model_selection.KFold(n_splits = 10, shuffle = True )

cv_results = model_selection.cross_val_score(clf, X_train, Y_train, cv=kfold)

print cv_results

print >> f1, cv_results

# print "creating model ..."
# clf.fit(X_train, Y_train)
# print "model created"

# print "predicting test data ..."
# prediction = clf.predict(X_test)

# accuracy = accuracy_score(Y_test, prediction)

# print accuracy