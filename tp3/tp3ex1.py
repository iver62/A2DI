import numpy as np
import matplotlib.pyplot as plt

sigma = 0.05

def datagen(n):
    X_train = np.random.rand(int(0.2*n),2)
    X_test = np.random.rand(int(0.8*n),2)
    c_train = np.zeros(len(X_train), dtype=int)
    c_test = np.zeros(len(X_test), dtype=int)
    
    for i in range(len(X_train)):
        d = (0.5*X_train[i][0] + X_train[i][1] - 0.75) / (np.sqrt(np.square(0.5) + 1))
        theta = np.exp(-(np.square(d) / 2*np.square(sigma)))
        r = np.random.random()
        if (-X_train[i][0]/2 + 0.75) <= X_train[i][1]:
            if (r > theta/2):
                c_train[i] = -1
            else:
                c_train[i] = 1
        else:
            if (r > theta/2):
                c_train[i] = 1
            else:
                c_train[i] = -1

    for i in range(len(X_test)):
        d = (0.5*X_test[i][0] + X_test[i][1] - 0.75) / (np.sqrt(np.square(0.5) + 1))
        theta = np.exp(-(np.square(d) / 2*np.square(sigma)))
        if (-X_test[i][0]/2 + 0.75) <= X_test[i][1]:
            if (r > theta/2):
                c_test[i] = -1
            else:
                c_test[i] = 1
        else:
            if (r > theta/2):
                c_test[i] = 1
            else:
                c_test[i] = -1

    return X_train, X_test, c_train, c_test
    
def aff_dataset(X_train, X_test, c_train, c_test):
    x_pos=list()
    y_pos=list()
    x_neg=list()
    y_neg=list()
    for i in range(len(X_train)):
        if c_train[i]==1:
            x_pos.append(X_train[i][0])
            y_pos.append(X_train[i][1])
        else:
            x_neg.append(X_train[i][0])
            y_neg.append(X_train[i][1])
    for i in range(len(X_test)):
        if c_test[i]==1:
            x_pos.append(X_test[i][0])
            y_pos.append(X_test[i][1])
        else:
            x_neg.append(X_test[i][0])
            y_neg.append(X_test[i][1])
    plt.plot(x_neg,y_neg,'or')
    plt.plot(x_pos,y_pos,'ob')

def ptrain_v2(X_train, c_train):
    err_train=list()
    thetas=list()
    theta = np.array([np.random.random() for i in range(3)])
    nb_iter=0
    while nb_iter < 100*len(X_train):
        i=0
        while i < len(X_train):
            x_plus = np.array([X_train[i][0], X_train[i][1], 1])
            if sign(np.vdot(theta, x_plus)) != c_train[i]:
                theta = theta + c_train[i]*x_plus
            i+=1
        nb_iter+=1
        thetas.append(theta)
    return thetas[np.argmin(err_train)]
    
def sign(x):
    if x >= 0:
        return 1
    return -1

#def get_test_err(n):
#    X_train, X_test, c_train, c_test = datagen(n)
#    theta = ptrain(X_train, c_train)
#    cpt = 0
#    for i in range(len(X_test)):
#        real = c_test[i]
#        pred = ptest(X_test[i], theta)
#        if pred != real:
#            cpt += 1
#    return cpt / len(X_test) * 100
#
#def ptest(x, theta):
#    x_plus = np.array([x[0], x[1], 1])
#    return sign(np.vdot(x_plus, theta))
    
X_train, X_test, c_train, c_test = datagen(300)
aff_dataset(X_train, X_test, c_train, c_test)
