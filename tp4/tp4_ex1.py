import numpy as np

from sklearn import cross_validation
from sklearn import datasets
data=datasets.load_digits()
X=data.data
X=X.T
c=data.target
d,n=X.shape

def extract(X, c, n1, n2):
    rows = np.arange(X.shape[0])
    cols = []
    for i in range(n):
        if (c[i] in [n1, n2]):
            cols.append(i)
    X1 = np.array(X[rows[:,np.newaxis], cols], dtype=np.int)
    c1 = np.array(c[cols], dtype=np.int)
    c1 = np.sign(c1-np.mean(c1))
    return X1, c1
    
def kfold_data(X, c, k):
    X_train_folds = ()
    c_train_folds = ()
    X_test_folds = ()
    c_test_folds = ()
    kf = cross_validation.KFold(X.shape[1], n_folds=k)
    kf = list(kf)
    rows = np.arange(X.shape[0])
    for i in range(k):
        X_train_folds += (X[rows[:,np.newaxis],kf[i][0]], )
        X_test_folds += (X[rows[:,np.newaxis],kf[i][1]], )
        c_train_folds += (np.array([c[j] for j in kf[i][0]]), )
        c_test_folds += (np.array([c[j] for j in kf[i][1]]), )
    return X_train_folds, c_train_folds, X_test_folds, c_test_folds

def ptrain_v2(X_train, c_train, nb_iter):
    theta = np.random.random((X_train.shape[0]+1, 1))
    best_theta = theta
    best_err_train = get_err_train(X_train, c_train, theta)
    print('err_train début', best_err_train)
    for n in range(nb_iter):
        for i in range(X_train.shape[1]):
            x_plus = np.concatenate((X_train[:,[i]],[[1]]), axis=0)
            if sign(np.vdot(theta, x_plus)) != c_train[i]:
                theta = theta + c_train[i]*x_plus
        err_train = get_err_train(X_train, c_train, theta)
        if err_train < best_err_train:
            best_theta = theta
            best_err_train = err_train
            print('err_train :', best_err_train)
        if err_train == 0:
            break
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
    x_plus = np.concatenate((x,[[1]]), axis=0)
    return sign(np.vdot(x_plus, theta))
    
def get_eval(X_test, c_test, theta):
    mat = np.zeros((2,2))
    n = len(c_test)
    err = 0
    for i in range(n):
        pred = ptest(X_test[:,[i]], theta)
        if c_test[i] == 1 and pred == 1:
            mat[0][0] += 1
        elif c_test[i] == -1 and pred == 1:
            mat[1][0] += 1
            err += 1
        elif c_test[i] == 1 and pred == -1:
            mat[0][1] += 1
            err += 1
        else:
            mat[1][1] += 1
    return err/n * 100, mat
    
def sign(x):
    return 1 if x >= 0 else -1
    
X1, c1 = extract(X, c, 5, 6)
X_train_folds, c_train_folds, X_test_folds, c_test_folds = kfold_data(X1, c1, 5)
theta = ptrain_v2(X_train_folds[2], c_train_folds[2], 100)
err=[]
for i in range(len(X_test_folds)):
    err_test, mat = get_eval(X_test_folds[i], c_test_folds[i], theta)
    err.append(err_test)
    print('pli n°',i+1,'->', err_test)
    print('matrice de confusion:\n', mat)
print('erreur de test moyenne : ', np.mean(err))