import scipy.io
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = scipy.io.loadmat('mnist_big.mat')


X_train = np.load("small_X_train.p")
Y_train = np.load("small_Y_train.p")

X_test = data['X_test'][1:2000,:]
Y_test = np.squeeze(data['Y_test'])[1:2000]



k = [1, 5, 10, 15, 20]
accuracy = []
number = []
for i in k:
	print "fajiowefae"
	neigh = KNeighborsClassifier(n_neighbors=i)
	neigh.fit(X_train, Y_train)
	prediction = neigh.predict(X_test)
	temp = accuracy_score(Y_test, prediction)
	accuracy.append(temp)
	number.append(i)

print "ahdfoiahfwhe"
print accuracy
plt.plot(number,accuracy,'r-')
plt.axis([0, 20, 0.5, 1.2])
plt.show()
