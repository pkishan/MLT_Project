import scipy.io
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
import matplotlib.pyplot as plt
import time
from sklearn.metrics import classification_report

data = scipy.io.loadmat('mnist_big.mat')

X_test = data['X_test']
Y_test = np.squeeze(data['Y_test'])

X_train = data['X_train']
Y_train = np.squeeze(data['Y_train'])

trees = [1, 5, 10, 15, 20, 30, 50]

accuracy = []
number = []

start_time = time.time() 

kfold = model_selection.KFold(n_splits = 10, shuffle = True )

for i in trees:
	start_new = time.time()
	clf = RandomForestClassifier(n_estimators=i)
	prediction = model_selection.cross_val_score(clf, X_train, Y_train, cv=kfold)
	temp = sum(prediction)/10;
	# print classification_report(Y_test, prediction)
	accuracy.append(temp)
	number.append(i)
	print("Number of K =  %s" %(i) )
	print("Accuracy =  %s" %(temp) )
	print("Time takn = %s" %(time.time() - start_new))
	print "-----------------------------------"


print("The time taken by the RandomForestClassifier algorithm is ( %s ) seconds" % (time.time() - start_time))

plt.plot(number,accuracy,'r-')
plt.axis([0, 50, 0.5, 1])
plt.show()

best = np.argmax(accuracy)

start = time.time()

clf_best = RandomForestClassifier(n_estimators=30)
clf_best.fit(X_train, Y_train)
prediction = clf_best.predict(X_test)
temp = accuracy_score(Y_test, prediction)
print("The accuracy on the test data by KNN is %s" %(temp))
print("The time taken to fit the best tree random forest is (%s) second" %(time.time() - start))
print classification_report(Y_test, prediction)