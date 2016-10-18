import scipy.io
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression



data = scipy.io.loadmat('mnist_big.mat')

X_test = data['X_test']
Y_test = np.squeeze(data['Y_test'])

X_train = data['X_train']
Y_train = np.squeeze(data['Y_train'])

import time

model = LogisticRegression()
model.fit(X_train, Y_train)

prediction = model.predict(X_test)

temp = accuracy_score(Y_test, prediction)



print "The accuracy of the logistic regression is %s" %(temp)

print classification_report(Y_test, prediction)


print("The time taken by the Logistic Regression algorithm is ( %s ) seconds" % (time.time() - start_time))
