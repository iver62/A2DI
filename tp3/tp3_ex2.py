import numpy as np
import matplotlib.pyplot as plt

sigma=0.2

def datagen(n):
    x=np.linspace(0, 5, n)
#    y=list()
#    x_train=list()
    x_train=np.empty(0)
    x_test=np.empty(0)
    y_train=np.empty(0)
    y_test=np.empty(0)
    for i in range(n):
        y=np.random.normal(2*x[i]+1, sigma)
#        y.append(np.random.normal(2*x[i]+1, sigma))
        if i%3 == 0:
            x_train=np.append(x[i])
            y_train=np.append(y)
        else:
            x_test=np.append(x[i])
            y_test=np.append(y)
    return x_train, x_test, y_train, y_test
    
def theta_star(X_train, Y_train):
#    print(np.vstack((X_train,np.ones(len(X_train)))))
    X=np.c_[X_train,np.ones(len(X_train))]
    g=np.linalg.inv(np.dot(X,X.T))
#    d=np.dot(X,Y_train)
    return np.dot(np.dot(g,X),Y_train)
    
    
X_train,X_test,Y_train,Y_test=datagen(90)
plt.figure()
plt.plot(X_train,Y_train,'.b',X_test,Y_test,'.b')
plt.figure()
plt.plot(X_train,Y_train,'.r',X_test,Y_test,'.b')
theta_star=theta_star(X_train,Y_train)