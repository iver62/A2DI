import numpy as np
import matplotlib.pyplot as plt

def datagen(n, fun=None):
    if fun == 'sin':
        x=np.linspace(-10, 10, n)
    else:
        x=np.linspace(0, 5, n)
    x_train=np.empty(0)
    x_test=np.empty(0)
    y_train=np.empty(0)
    y_test=np.empty(0)
    for i in range(n):
        if fun == 'sin':
            y=np.random.normal(np.sin(x[i])/x[i], 0.05)
        else:
            y=np.random.normal(2*x[i]+1, 0.2)
#        y.append(np.random.normal(2*x[i]+1, sigma))
        if i%3 == 0:
            x_train=np.append(x_train,x[i])
            y_train=np.append(y_train,y)
        else:
            x_test=np.append(x_test,x[i])
            y_test=np.append(y_test,y)
    return x_train, x_test, y_train, y_test
    
#def datagen_v2(n):
#    x=np.linspace(-10, 10, n)
#    x_train=np.empty(0)
#    x_test=np.empty(0)
#    y_train=np.empty(0)
#    y_test=np.empty(0)
#    for i in range(n):
#        y=np.random.normal(np.sin(x[i])/x[i], 0.05)
#        if i%3 == 0:
#            x_train=np.append(x_train,x[i])
#            y_train=np.append(y_train,y)
#        else:
#            x_test=np.append(x_test,x[i])
#            y_test=np.append(y_test,y)
#    return x_train, x_test, y_train, y_test
    
def theta_star(X_train, Y_train):
    X=np.vstack((X_train,np.ones(len(X_train))))
    g=np.linalg.inv(np.dot(X,X.T))
    return np.dot(np.dot(g,X),Y_train)
    
def polyreg(X_train, Y_train, d):
    X=np.vstack((X_train,np.ones(len(X_train))))
    g=np.linalg.inv(np.dot(X,X.T))
    return np.dot(np.dot(g,X),Y_train)

X_train,X_test,Y_train,Y_test=datagen(90)
plt.figure()
plt.plot(X_train,Y_train,'.r',X_test,Y_test,'.b')
theta_star=theta_star(X_train,Y_train)
plt.plot([theta_star[0]*x + theta_star[1] for x in range(6)],'k')
X_train,X_test,Y_train,Y_test=datagen(90,fun='sin')
plt.figure()
plt.plot(X_train,Y_train,'.r',X_test,Y_test,'.b')