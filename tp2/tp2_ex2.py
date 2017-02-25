import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
#import sys
#sys.path.append('C:/Users/Pierrick/Desktop/A2DI/tp1')
#from tp1_ex2 import ptrain, ptest

def datagen(n):
    X = np.random.random((2, n))
    c = np.zeros(n, dtype=int)
    for i in range(n):
        if (-X[0][i]/2 + 0.75) <= X[1][i]:
            c[i] = 1
        else:
            c[i] = -1
    return train_test_split(X.T, c, test_size=0.9)
    
def get_test_err(n):
    X_train, X_test, c_train, c_test = datagen(n)
    X_train = X_train.T
    X_test = X_test.T
    theta = ptrain(X_train, c_train)
    cpt = 0
    for i in range(X_test.shape[1]):
        real = c_test[i]
        pred = ptest(X_test[:,[i]], theta)
        if pred != real:
            cpt += 1
    return cpt / X_test.shape[1] * 100

def ptrain(X_train, c_train):
    theta = np.random.random((3,1))
    i = 0
    while i < X_train.shape[1]:
        x_plus=np.concatenate((X_train[:,[i]],[[1]]), axis=0)
        if sign(np.vdot(theta, x_plus)) == c_train[i]:
            i += 1
        else:
            theta = theta + c_train[i]*x_plus
            i = 0
    return theta
    
def sign(x):
    if x >= 0:
        return 1
    return -1
    
def ptest(x, theta):
    x_plus=np.concatenate((x,[[1]]),axis=0)
    return sign(np.vdot(x_plus, theta))


err=[]
maxi=20
bin_size=0.5
bins=np.arange(0.0,np.ceil(maxi+1),bin_size)
bin_centers=bins[:-1]+bin_size/2
#crit=1.0
perr=np.ones(bin_centers.shape)
#count=0
#while crit < 0.001:
for i in range(20):
    err.append(get_test_err(445))
#perr_old=perr
(perr, bins_out) = np.histogram(err, bins=bins, density=True)
#crit = np.max(np.abs(perr-perr_old))
#count += 1
    
plt.figure()
plt.bar(bins[:-1], perr, width=bin_size)
plt.xlabel('test error rate')
plt.title('n=445')

EX1 = np.sum(np.multiply(perr,bin_centers))
print('Erreur de généralisation :', EX1)