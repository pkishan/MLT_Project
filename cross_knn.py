import scipy.io
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time
from sklearn.metrics import classification_report


data = scipy.io.loadmat('mnist_big.mat')

X_test = data['X_test']
Y_test = np.squeeze(data['Y_test'])

X_train = data['X_train']
Y_train = np.squeeze(data['Y_train'])

k = [1, 5, 10, 15, 20]
accuracy = []
number = []

start_time = time.time()

for i in k:
	neigh = KNeighborsClassifier(n_neighbors=i)
	neigh.fit(X_train, Y_train)
	prediction = neigh.predict(X_test)
	temp = accuracy_score(Y_test, prediction)
	print classification_report(Y_test, prediction)
	accuracy.append(temp)
	number.append(i)

print("The time taken by the knn algorithm is ( %s ) seconds" % (time.time() - start_time))

plt.plot(number,accuracy,'r-')
# plt.axis([0, 20, 0.5, 1])
plt.show()