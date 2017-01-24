import numpy as np

from sklearn import datasets
data=datasets.load_digits()
X=data.data
X=X.T
c=data.target
d,n=X.shape

def extract(X, c):
    X1 = np.array([X[:,i] for i in range(n) if c[i]==5 or c[i]==6], dtype=int)
    c1 = np.array([c[i] for i in range(n) if c[i]==5 or c[i]==6], dtype=int)
    c1 = np.sign(c1-np.mean(c1))
    return X1.T, c1
    
def kfold_data(X, c, k):
    X_train_folds = ()
    c_train_folds = ()
    X_test_folds = ()
    c_test_folds = ()
    n = X.shape[1]
    for i in range(1,k+1):
        X_train_folds += (X[:,int(n*(k-i)/k):int(n*(k-i+1)/k)], )
        c_train_folds += (c[int(n*(k-i)/k):int(n*(k-i+1)/k)], )
        X_test_folds += (np.concatenate((X[:,:int(n*(k-i)/k)],X[:,int(n*(k-i+1)/k):]),axis=1), )
        c_test_folds += (np.concatenate((c[:int(n*(k-i)/k)],c[int(n*(k-i+1)/k):])), )
    return X_train_folds, c_train_folds, X_test_folds, c_test_folds

def ptrain_v2(X_train, c_train, nb_iter):
    theta = np.random.random((len(X_train)+1,1))
    best_theta = theta
    best_err_train = get_err_train(X_train, c_train, theta)
    print('err_train',best_err_train)
    for n in range(nb_iter):
        for i in range(X_train.shape[1]):
            x_plus = np.concatenate((X_train[:,[i]],[[1]]), axis=0)
            if sign(np.vdot(theta, x_plus)) != c_train[i]:
                theta = theta + c_train[i]*x_plus
        err_train = get_err_train(X_train, c_train, theta)
        if err_train < best_err_train:
            best_theta = theta
            best_err_train = err_train
            print('err_train :',best_err_train)
        if err_train == 0:
            break
    return best_theta
    
def get_err_train(X_train, c_train, theta):
    err=[]
    for i in range(len(c_train)):
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
    
def get_eval(X_test, c_test, theta):
    mat = np.zeros((2,2))
    n = len(c_test)
    err=0
    for i in range(n):
        if c_test[i] == 1 and ptest(X_test[:,[i]],theta) == 1:
            mat[0][0] += 1
        elif c_test[i] == -1 and ptest(X_test[:,[i]],theta) == 1:
            mat[1][0] += 1
            err += 1
        elif c_test[i] == 1 and ptest(X_test[:,[i]],theta) == -1:
            mat[0][1] += 1
            err += 1
        else:
#        if c_test[i] == 1 and ptest(X_test[:,[i]],c_test[i],theta) == 1:
            mat[1][1] += 1
#        err += loss(X_test[:,[i]],c_test[i],theta)
    return err/n * 100, mat
    
def sign(x):
    if x >= 0:
        return 1
    return -1
    
X1,c1 = extract(X,c)
X_train_folds,c_train_folds,X_test_folds,c_test_folds=kfold_data(X1,c1,5)
theta = ptrain_v2(X_train_folds[0], c_train_folds[0], 1000)
err=[]
for i in range(len(X_test_folds)):
    err_test, mat = get_eval(X_test_folds[i], c_test_folds[i], theta)
    err.append(err_test)
    print('pli n°',i+1,'->',err_test)
    print('matrice de confusion:\n',mat)
print('erreur de test moyenne :',np.mean(err))
print('L\'erreur de test est variable d\'un pli à l\'autre mais elle ne change pas d\'une exécution à une autre')