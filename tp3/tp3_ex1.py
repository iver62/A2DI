import numpy as np
import matplotlib.pyplot as plt

def datagen(n):
    X = np.random.random((2,n))
    c = np.zeros(n, dtype=int)
    for i in range(X.shape[1]):
        d = np.abs(-0.5*X[0][i] - X[1][i] + 0.75) / (np.sqrt(np.square(0.5) + 1))
        theta = np.exp(-(np.square(d) / (2*np.square(0.05))))
        mistake = np.random.binomial(1,theta/2)
        if (-X[0][i]/2 + 0.75) <= X[1][i]:
            c[i] = 1
        else:
            c[i] = -1
        if mistake == 1:
            c[i] = -c[i]
    return X[:, :int(0.2*n)], X[:, int(0.2*n):], c[:int(0.2*n)], c[int(0.2*n):]

def ptrain_v2(X_train, c_train, nb_iter):
    theta = np.random.random((3,1))
    best_theta = theta
    best_err_train = get_err_train(X_train, c_train, theta)
    for n in range(nb_iter):
        for i in range(X_train.shape[1]):
            x_plus = np.concatenate((X_train[:,[i]],[[1]]), axis=0)
            if sign(np.vdot(theta, x_plus)) != c_train[i]:
                theta = theta + c_train[i]*x_plus
        err_train = get_err_train(X_train, c_train, theta)
        if err_train < best_err_train:
            best_theta = theta
            best_err_train = err_train
        print('err_train :',get_err_train(X_train, c_train, theta))
    return best_theta
    
def get_err_train(X_train, c_train, theta):
    err=[]
    for i in range(X_train.shape[1]):
        err.append(loss(X_train[:,[i]], c_train[i], theta))
    return np.mean(err)*100
    
def loss(x, c, theta):
    pred = ptest(x, theta)
    if c == pred:
        return 0
    return 1

def ptest(x, theta):
    x_plus = np.concatenate((x,[[1]]),axis=0)
    return sign(np.vdot(x_plus, theta))
    
def sign(x):
    if x >= 0:
        return 1
    return -1

def aff_dataset(X_train, X_test, c_train, c_test):
    X = np.concatenate((X_train,X_test),axis=1)
    c = np.concatenate((c_train,c_test))
    x_pos=[]
    y_pos=[]
    x_neg=[]
    y_neg=[]
    for i in range(X.shape[1]):
        if c[i] == 1:
            x_pos.append(X[0][i])
            y_pos.append(X[1][i])
        else:
            x_neg.append(X[0][i])
            y_neg.append(X[1][i])
    plt.plot(x_neg,y_neg,'.r',x_pos,y_pos,'.b')

    
X_train, X_test, c_train, c_test = datagen(300)
aff_dataset(X_train, X_test, c_train, c_test)
theta = ptrain_v2(X_train, c_train, 100)
plt.plot([(-theta[0]*x-theta[2]) / theta[1] for x in range(2)], 'k')