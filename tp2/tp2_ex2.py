import numpy as np
import matplotlib.pyplot as plt
#from . tp1.tp1_ex2 import ptrain, ptest

def datagen(n):
    X = np.random.random((2,n))
    c = np.zeros(n, dtype=int)
    for i in range(X.shape[1]):
        if (-X[0][i]/2 + 0.75) <= X[1][i]:
            c[i] = 1
        else:
            c[i] = -1
    return X[:, :int(0.1*n)], X[:, int(0.1*n):], c[:int(0.1*n)], c[int(0.1*n):]
    
def get_test_err(n):
    X_train, X_test, c_train, c_test = datagen(n)
    theta = ptrain(X_train, c_train)
    cpt = 0
    for i in range(X_test.shape[1]):
        real = c_test[i]
        pred = ptest(X_test[:,[i]], theta)
        if pred != real:
            cpt += 1
    return cpt / X_test.shape[1] * 100

def ptrain(X_train, c_train):
    theta=np.random.random((3,1))
    i=0
    while i < X_train.shape[1]:
        x_plus=np.concatenate((X_train[:,[i]],[[1]]),axis=0)
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


err=list()
maxi=25
bin_size=0.5
bins=np.arange(0.0,np.ceil(maxi+1),bin_size)
bin_centers=bins[:-1]+bin_size/2
#crit=1.0
perr=np.ones(bin_centers.shape)
#count=0
#while crit < 0.001:
for i in range(20):
    err.append(get_test_err(445))
perr_old=perr
(perr,bins_out) = np.histogram(err, bins=bins, normed=True)
#crit = np.max(np.abs(perr-perr_old))
#count += 1
    
plt.figure()
plt.bar(bins[:-1], perr, width=bin_size)
plt.xlabel('test error rate')
plt.title('n=445')

EX1 = np.sum(np.multiply(perr,bin_centers))
print('Erreur de généralisation :', EX1)