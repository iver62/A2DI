import numpy as np
import matplotlib.pyplot as plt

def datagen(n, fun=None):
    if fun == 'sin':
        x = np.linspace(-10, 10, n)
    else:
        x = np.linspace(0, 5, n)
    X_train = np.empty(0)
    X_test = np.empty(0)
    Y_train = np.empty(0)
    Y_test = np.empty(0)
    for i in range(n):
        if fun == 'sin':
            y = np.random.normal(np.sin(x[i])/x[i], 0.05)
        else:
            y = np.random.normal(2*x[i]+1, 0.2)
        if i%3 == 0:
            X_train = np.append(X_train, x[i])
            Y_train = np.append(Y_train, y)
        else:
            X_test = np.append(X_test, x[i])
            Y_test = np.append(Y_test, y)
    return X_train, X_test, Y_train, Y_test
    
def theta_star(X_train, Y_train):
    X_train = np.vstack((X_train, np.ones(X_train.shape[0])))
    g = np.linalg.inv(np.dot(X_train, X_train.T))
    return np.dot(np.dot(g, X_train), Y_train)
    
def polyreg(X_train, Y_train, p):
    n = X_train.shape[0]
    X = np.ones((p+1, n))
    for i in range(n):
        for j in range(p+1):
            X[j, i] = np.power(X_train[i], p-j)
#    return theta_star(X, Y_train)
    g = np.linalg.inv(np.dot(X, X.T))
    return np.dot(np.dot(g, X), Y_train)
    
def f(x, theta_star):
    return np.vdot(theta_star, x)
#    d = theta_star.shape[0]
#    return np.sum([theta_star[j]*np.power(x, d-j-1) for j in range(d)])
    
def get_err_train(X_train, Y_train, p):
    theta_star = polyreg(X_train, Y_train, p)
    err = []
    for i in range(X_train.shape[0]):
        err.append(loss(X_train[i], Y_train[i], theta_star))
    return np.sum(err)

#def get_err_train(X_train, Y_train):
#    absc=np.arange(1,18)
#    ords=[]
#    for p in range(1,18):
#        ords.append(err_train(X_train, Y_train, p))
#    return absc,ords
    
#def loss(X,Y,p):
#    theta_star=polyreg(X,Y,p)
#    err=[]
#    for i in range(len(X)):
#        err.append(np.square(Y[i]-f(X[i],theta_star)))
#    return np.squre()
    
def loss(x, y, theta):
    return np.square(y - f(x, theta))
    
def aff_courbe(theta):
    absc=np.arange(-10,10,0.1)
    ords=np.zeros(len(absc))
    for i in range(len(absc)):
        ords[i]=f(absc[i], theta)
#        ords[i]=np.sum([theta[p]*np.power(absc[i],len(theta)-p-1) for p in range(len(theta))])
    plt.plot(absc,ords,'k-')

X_train, X_test, Y_train, Y_test = datagen(90)
plt.figure()
plt.plot(X_train, Y_train, '.r', X_test, Y_test, '.b')
theta_star = theta_star(X_train, Y_train)
#aff_courbe(theta_star)
plt.plot([theta_star[0]*x + theta_star[1] for x in range(6)], 'k')
X_train, X_test, Y_train, Y_test = datagen(90,fun='sin')
plt.figure()
plt.plot(X_train, Y_train, '.r', X_test, Y_test, '.b')
theta_star = polyreg(X_train, Y_train, 8)
aff_courbe(theta_star)
plt.figure()
absc = np.arange(1, 18)
train_ords = []
test_ords = []
for p in range(1, 18):
    train_ords.append(get_err_train(X_train, Y_train, p))
#train_absc,train_ords=get_train_err(X_train,Y_train)
#test_absc,test_ords=get_train_err(X_test,Y_test)
plt.plot(absc, train_ords, 'g-')
theta_star = polyreg(X_train, Y_train, 19)
plt.figure()
plt.plot(X_train, Y_train, '.r')
aff_courbe(theta_star)

