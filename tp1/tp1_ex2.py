import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

def datagen(n, d):
    X = np.random.random((d, n))
    c = np.zeros(n, dtype=int)
    for i in range(n):
        if (-X[0][i]/2 + 0.75) <= X[1][i]:
            c[i] = 1
        else:
            c[i] = -1
    return X, c
    
def split(D):
    return D[:,:int(0.8*len(D[0]))], D[:,int(0.8*len(D[0])):]
    
def ptrain(X_train, c_train):
    theta = np.random.random((3,1))
    i = 0
    while i < X_train.shape[1]:
        x_plus=np.concatenate((X_train[:,[i]],[[1]]), axis=0)
        if sign(np.vdot(theta, x_plus)) == c_train[i]:
            i += 1
        else:
            theta=theta + c_train[i]*x_plus
            i = 0
    return theta
    
def ptest(x, theta):
    x_plus=np.concatenate((x,[[1]]),axis=0)
    return sign(np.vdot(x_plus, theta))

def get_test_err(n, d):
    X, c = datagen(n, d)
    X_train, X_test, c_train, c_test = train_test_split(X.T, c, test_size=0.2)
    X_train = X_train.T
    X_test = X_test.T
    theta = ptrain(X_train, c_train)
    aff_dataset(X, c, theta)
    cpt = 0
    for i in range(X_test.shape[1]):
        real = c_test[i] #classe réelle
        pred = ptest(X_test[:,[i]],theta) #classe prédite
        if pred != real:
            cpt += 1
    return cpt/X_test.shape[1]*100

def sign(x):
    if x >= 0:
        return 1
    return -1

def aff_dataset(X, c, theta):
    n = X.shape[1]
    x_pos=[X[0][i] for i in range(n) if c[i]==1]
    y_pos=[X[1][i] for i in range(n) if c[i]==1]
    x_neg=[X[0][i] for i in range(n) if c[i]==-1]
    y_neg=[X[1][i] for i in range(n) if c[i]==-1]
    plt.figure()
    plt.plot(x_neg,y_neg,'.r',x_pos,y_pos,'.b')
    plt.plot([(-theta[0]*x - theta[2]) / theta[1] for x in range(2)], 'k')

err=[]
for i in range(20):
    print('i=',i)
    err.append(get_test_err(100,2))
print('Erreur de généralisation :',np.mean(err),'%')