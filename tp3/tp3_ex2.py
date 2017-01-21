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
    
def polyreg(X_train, Y_train, p):
    X=np.ones((p+1,len(X_train)))
    for i in range(len(X_train)):
        for j in range(p+1):
            X[j,i]=np.power(X_train[i],p-j)
    g=np.linalg.inv(np.dot(X,X.T))
    return np.dot(np.dot(g,X),Y_train)
    
def f(x,theta_star):
    return np.sum([theta_star[j]*np.power(x,len(theta_star)-j-1) for j in range(len(theta_star))])
    
def err_train(X_train,Y_train,p):
    theta_star=polyreg(X_train,Y_train,p)
    err=[]
    for i in range(len(X_train)):
        err.append(loss(X_train[i],Y_train[i],theta_star))
    return np.sum(err)

def get_train_err(X_train,Y_train):
    absc=np.arange(1,18)
    ords=[]
    for p in range(1,18):
        ords.append(err_train(X_train,Y_train,p))
    return absc,ords
    
#def loss(X,Y,p):
#    theta_star=polyreg(X,Y,p)
#    err=[]
#    for i in range(len(X)):
#        err.append(np.square(Y[i]-f(X[i],theta_star)))
#    return np.squre()
    
def loss(x,y,theta):
#    pred=np.sum([theta[j]*np.power(x,len(theta)-j-1) for j in range(len(theta))])
    return np.square(y-f(x,theta))
#    return np.abs(y-pred)
    
def aff_courbe(theta):
    absc=np.arange(-10,10,0.1)
    ords=np.zeros(len(absc))
    for i in range(len(absc)):
        ords[i]=f(absc[i],theta)
#        ords[i]=np.sum([theta[p]*np.power(absc[i],len(theta)-p-1) for p in range(len(theta))])
    plt.plot(absc,ords,'k-')
    
X_train,X_test,Y_train,Y_test=datagen(90)
plt.figure()
plt.plot(X_train,Y_train,'.r',X_test,Y_test,'.b')
theta_star=theta_star(X_train,Y_train)
#aff_courbe(theta_star)
plt.plot([theta_star[0]*x + theta_star[1] for x in range(6)],'k')
X_train,X_test,Y_train,Y_test=datagen(90,fun='sin')
plt.figure()
plt.plot(X_train,Y_train,'.r',X_test,Y_test,'.b')
theta_star=polyreg(X_train,Y_train,8)
aff_courbe(theta_star)
plt.figure()
train_absc,train_ords=get_train_err(X_train,Y_train)
test_absc,test_ords=get_train_err(X_test,Y_test)
plt.plot(train_absc,train_ords,'g-',test_absc,test_ords,'b-')

