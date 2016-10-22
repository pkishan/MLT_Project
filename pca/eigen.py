import scipy.io
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
f1 = open("eigen_analysis.txt","w+")

data = scipy.io.loadmat('../mnist_big.mat')



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
# print >> f1, S


S = np.multiply((1.0/N),S)

# print >> f1,  S
w,v = LA.eig(S)
np.sort(w)
w = w / w.max(axis=0)
# w = LA.norm(w)

plt.plot(w)
# plt.axis([0, 800, 0, 40000])
plt.show()


dimen = [10, 50, 100, 200, 400, 500, 600, 783]
area = []
sum = 0

for i in range(0,784):
	sum = sum + w[i];
	if i in dimen :
		area.append(sum)

area = area / max(area)

print >>f1, "dimensions = ",dimen,"\n","Area =", area

plt.plot(dimen,area,'r-')
plt.show()


plt.plot(w)
plt.axis([0, 200, 0, 1])
plt.show()


# print w