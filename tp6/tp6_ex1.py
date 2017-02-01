import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
#from mpl_toolkits.mplot3d import Axes3D
    
def datagen(n):
    X = np.random.random((2,n))
    c = np.zeros(n, dtype=int)
    for i in range(n):
        d = np.abs(-0.5*X[0][i] - X[1][i] + 0.75) / (np.sqrt(np.square(0.5) + 1))
        r = np.exp(-(np.square(d) / (2*np.square(0.05))))
        mistake = np.random.binomial(1,r/2)
        if (-X[0][i]/2 + 0.75) <= X[1][i]:
            c[i] = 1
        else:
            c[i] = 0
        if mistake == 1:
            c[i] = 1-c[i]
    return X, c

def aff_dataset(X, c):
    n = X.shape[1]
    x_pos=[X[0][i] for i in range(n) if c[i] == 1]
    y_pos=[X[1][i] for i in range(n) if c[i] == 1]
    x_neg=[X[0][i] for i in range(n) if c[i] == 0]
    y_neg=[X[1][i] for i in range(n) if c[i] == 0]
    plt.plot(x_neg,y_neg,'.r',x_pos,y_pos,'.b')
      
def fold_data(X, c):
    X_train, X_test, c_train, c_test = train_test_split(X.T, c, test_size=0.2)
    return X_train.T, X_test.T, c_train, c_test
    
def batch(X_train, c_train, eta):
    d,n = X_train.shape
    X_plus = np.vstack((X_train, np.ones(n)))
    thetas = np.zeros((d+1,526))
    theta = np.random.random(d+1)
    cpt = 0
    while cpt<526:
        err = np.zeros(n)
        for i in range(n):
            pred = sgm(np.dot(X_plus[:,i].T,theta))
            err[i] = pred - c_train[i]
        g = np.dot(X_plus, err)
#        print(g)
        theta -= eta * g
#        print(theta)
        thetas[:,cpt] = theta
        cpt += 1
    return thetas

def sgm(z):
    return 1 / (1 + np.exp(-z))
    
def visualize(thetas):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(thetas[0],thetas[1],thetas[2],'-o')
    plt.draw()

X, c = datagen(300)        
aff_dataset(X, c)
X_train, X_test, c_train, c_test = fold_data(X, c)
#X_plus = np.vstack((X, np.ones(X.shape[1])))
thetas = batch(X_train, c_train, 0.02)
visualize(thetas)