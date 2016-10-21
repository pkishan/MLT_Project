import scipy.io
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
f1 = open("dsd.txt","w+")

data = scipy.io.loadmat('mnist_big.mat')



Xtrain = data['X_train']

N, D = Xtrain.shape

# print Xtrain.shape 

Xmean = np.mean(Xtrain, axis = 0)

Xmean = np.asarray([Xmean])
# print Xmean.shape

# print np.repeat(Xmean, N, axis = 0), np.repeat(Xmean, N, axis = 0 ).shape 

X = Xtrain - np.repeat(Xmean, N, axis = 0 )
 
print Xtrain.shape

S = np.dot(np.transpose(X),X)
print >> f1, S
S = np.multiply((1.0/N),S)

print >> f1,  S
w,v = LA.eig(S)
# w = LA.norm(w)

plt.plot(w)
# plt.axis([0, 800, 0, 40000])
plt.show()

plt.plot(w)
plt.axis([40, 200, 0, 5000])
plt.show()

print w