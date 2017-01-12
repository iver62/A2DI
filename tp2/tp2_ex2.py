import numpy as np
import matplotlib.pyplot as plt

def datagen(n):
    X_train = np.zeros(shape=(int(0.1*n),2))
    X_test = np.zeros(shape=(int(0.9*n),2))
    c_train = np.zeros(int(0.1*n), dtype=int)
    c_test = np.zeros(int(0.9*n), dtype=int)
    for i in range(int(n*0.1)):
        p = [np.random.random(), np.random.random()]
        X_train[i] = p
        if (-p[0]/2 + 0.75) <= p[1]:
            c_train[i] = 1
        else:
            c_train[i] = -1
    for i in range(int(n*0.9)):
        p = [np.random.random(), np.random.random()]
        X_test[i] = p
        if (-p[0]/2 + 0.75) <= p[1]:
            c_test[i] = 1
        else:
            c_test[i] = -1
    return X_train, X_test, c_train, c_test
    
def get_test_err(n):
    X_train, X_test, c_train, c_test = datagen(n)
    theta = ptrain(X_train, c_train)
    cpt = 0
    for i in range(len(X_test)):
        real = c_test[i]
        pred = ptest(X_test[i], theta)
        if pred != real:
            cpt += 1
    return cpt / len(X_test) * 100
    
 
def ptrain(X_train, c_train):
    theta = np.array([np.random.random() for i in range(3)])
    i=0
    while i < len(X_train):
        x_plus = np.array([X_train[i][0], X_train[i][1], 1])
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
    x_plus = np.array([x[0], x[1], 1])
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
for i in range(100):
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
print('Erreur de généralisation : ', EX1)