import numpy as np
import matplotlib.pyplot as plt

def datagen(n):
    X_train=np.random.rand(int(0.2*n),2)
    X_test=np.random.rand(int(0.8*n),2)
    c_train=np.empty(0, dtype=int)
    c_test=np.empty(0, dtype=int)
    for i in range(len(X_train)):
        if (-X_train[i][0]/2 + 0.75) <= X_train[i][1]:
            c_train=np.append(c_train,1)
        else:
            c_train=np.append(c_train,-1)
    for i in range(len(X_test)):
        if (-X_test[i][0]/2 + 0.75) <= X_test[i][1]:
            c_test=np.append(c_test,1)
        else:
            c_test=np.append(c_test,-1)
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
    theta=np.random.rand(3)
    i=0
    while i < len(X_train):
        x_plus=np.append(X_train[i],1)
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
    x_plus=np.append(x,1)
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
#    print(i)
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