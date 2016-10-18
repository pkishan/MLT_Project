# --------------------------
# Using Recursive feature elimination to train svm on small data
# Wikipedia : Recursive Feature Elimination algorithm,
# 			  commonly used with Support Vector Machines 
#			  to repeatedly construct a model and remove features with low weights.
# --------------------------
import numpy as np
import scipy.io
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

data = scipy.io.loadmat('mnist_big.mat')

X_train = np.load("small_X_train.p")
Y_train = np.load("small_Y_train.p")

X_test = data['X_test']
Y_test = np.squeeze(data['Y_test'])

clf = svm.LinearSVC()  # works with linear kernel only

# print "creating model ..."
# selector.fit(X_train, Y_train)
# print "model created"

# print "predicting test data ..."
# prediction = selector.predict(X_test)
# print prediction

# accuracy = accuracy_score(Y_test, prediction)

# print accuracy

k = [5, 10, 100, 200, 400, 500, 600]
accuracy = []
number = []
for i in k:
	
	selector = RFE(clf,i)
	print "creating model for i = ", i
	selector.fit(X_train, Y_train)
	prediction = selector.predict(X_test)
	temp = accuracy_score(Y_test, prediction)
	accuracy.append(temp)
	number.append(i)

print "accuracy ="
print accuracy
plt.plot(number,accuracy,'r-')

plt.show()