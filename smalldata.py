import scipy.io
from sklearn.svm import SVC
import numpy as np

bigdata = scipy.io.loadmat('mnist_big.mat')

X_train = bigdata['X_train']
Y_train = bigdata['Y_train']


X_test = bigdata['X_train']
Y_test = bigdata['X_train']

print X_train.shape
print X_train
print "----------------"
print Y_train.shape
print Y_train
print "----------------"

sliced_X_train = X_train[0:100,:]
print sliced_X_train.shape

small_X_train = X_train[2000:3000,:]
small_Y_train = Y_train[2000:3000,:]

# --------------------------------------

X_train_temp =  X_train[8000:9000,:]
Y_train_temp =  Y_train[8000:9000,:]

small_X_train = np.vstack((small_X_train,X_train_temp))
small_Y_train = np.vstack((small_Y_train,Y_train_temp))
# --------------------------------------

# --------------------------------------

X_train_temp =  X_train[15000:16000,:]
Y_train_temp =  Y_train[15000:16000,:]

small_X_train = np.vstack((small_X_train,X_train_temp))
small_Y_train = np.vstack((small_Y_train,Y_train_temp))
# --------------------------------------

# --------------------------------------

X_train_temp =  X_train[21000:22000,:]
Y_train_temp =  Y_train[21000:22000,:]

small_X_train = np.vstack((small_X_train,X_train_temp))
small_Y_train = np.vstack((small_Y_train,Y_train_temp))
# --------------------------------------

# --------------------------------------

X_train_temp =  X_train[27000:28000,:]
Y_train_temp =  Y_train[27000:28000,:]

small_X_train = np.vstack((small_X_train,X_train_temp))
small_Y_train = np.vstack((small_Y_train,Y_train_temp))
# --------------------------------------

# --------------------------------------

X_train_temp =  X_train[33000:34000,:]
Y_train_temp =  Y_train[33000:34000,:]

small_X_train = np.vstack((small_X_train,X_train_temp))
small_Y_train = np.vstack((small_Y_train,Y_train_temp))
# --------------------------------------

# --------------------------------------

X_train_temp =  X_train[39000:40000,:]
Y_train_temp =  Y_train[39000:40000,:]

small_X_train = np.vstack((small_X_train,X_train_temp))
small_Y_train = np.vstack((small_Y_train,Y_train_temp))
# --------------------------------------

# --------------------------------------

X_train_temp =  X_train[45000:46000,:]
Y_train_temp =  Y_train[45000:46000,:]

small_X_train = np.vstack((small_X_train,X_train_temp))
small_Y_train = np.vstack((small_Y_train,Y_train_temp))
# --------------------------------------

# --------------------------------------

X_train_temp =  X_train[51000:52000,:]
Y_train_temp =  Y_train[51000:52000,:]

small_X_train = np.vstack((small_X_train,X_train_temp))
small_Y_train = np.vstack((small_Y_train,Y_train_temp))
# --------------------------------------

# --------------------------------------

X_train_temp =  X_train[57000:58000,:]
Y_train_temp =  Y_train[57000:58000,:]

small_X_train = np.vstack((small_X_train,X_train_temp))
small_Y_train = np.vstack((small_Y_train,Y_train_temp))
# --------------------------------------

small_Y_train = small_Y_train.flatten()

print small_Y_train.shape , small_X_train.shape

small_X_train.dump("small_X_train.p")
small_Y_train.dump("small_Y_train.p")


# clf = SVC()
# clf.fit(X_train, Y_train)

# Y_pred = clf.predict(Y_test)

