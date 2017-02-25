import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

def datagen(n):
    X = np.random.random((2, n))
    c = np.zeros(n, dtype=int)
    for i in range(n):
        d = np.abs(-0.5*X[0][i] - X[1][i] + 0.75) / (np.sqrt(np.square(0.5) + 1))
        theta = np.exp(-(np.square(d) / (2*np.square(0.05))))
        mistake = np.random.binomial(1, theta/2)
        if (-X[0][i]/2 + 0.75) <= X[1][i]:
            c[i] = 1
        else:
            c[i] = -1
        if mistake == 1:
            c[i] = -c[i]
    return X, c

def ptrain_v2(X_train, c_train, X_test, c_test, nb_iter):
    theta = np.random.random((X_train.shape[0]+1,1))
    best_theta = theta
    best_err_train = get_err_train(X_train, c_train, theta)
    for n in range(nb_iter):
        for i in range(X_train.shape[1]):
            x_plus = np.concatenate((X_train[:,[i]],[[1]]), axis=0)
            if sign(np.vdot(theta, x_plus)) != c_train[i]:
                theta = theta + c_train[i]*x_plus
        err_train = get_err_train(X_train, c_train, theta)
        err_test = get_err_train(X_test, c_test, theta)
        if err_train < best_err_train:
            best_theta = theta
            best_err_train = err_train
        print('iter n°', n, ' err_train :', err_train, 'err_test: ', err_test)
    return best_theta
    
def get_err_train(X_train, c_train, theta):
    err = []
    for i in range(X_train.shape[1]):
        err.append(loss(X_train[:,[i]], c_train[i], theta))
    return np.mean(err)*100
    
def loss(x, c, theta):
    pred = ptest(x, theta)
    return 0 if c == pred else 1

def ptest(x, theta):
    x_plus = np.concatenate((x,[[1]]),axis=0)
    return sign(np.vdot(x_plus, theta))
    
def sign(x):
    return 1 if x >= 0 else -1

def aff_dataset(X, c, theta):
    n = X.shape[1]
    x_pos = [X[0][i] for i in range(n) if c[i]==1]
    y_pos = [X[1][i] for i in range(n) if c[i]==1]
    x_neg = [X[0][i] for i in range(n) if c[i]==-1]
    y_neg = [X[1][i] for i in range(n) if c[i]==-1]
    plt.plot(x_neg,y_neg,'.r',x_pos,y_pos,'.b')
    plt.plot([(-theta[0]*x - theta[2]) / theta[1] for x in range(2)], 'k')

X, c = datagen(300)
X_train, X_test, c_train, c_test = train_test_split(X.T, c, test_size=0.8)
X_train = X_train.T
X_test = X_test.T
theta = ptrain_v2(X_train, c_train, X_test, c_test, 100)
aff_dataset(X, c, theta)