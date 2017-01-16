import numpy as np

from sklearn import datasets
data=datasets.load_digits()
X=data.data
X=X.T
c=data.target
d,n=X.shape

def extract(X, c):
    X1=np.array([X[:,i] for i in range(len(c)) if c[i]==5 or c[i]==6])
    c1=np.array([c[i] for i in range(len(c)) if c[i]==5 or c[i]==6])
    c1=np.sign(c1-np.mean(c1))
    return X1.T, c1
    
def kfold_data(X, c, k):
#    size = len(X)/k
    X_train_folds=tuple()
    c_train_folds=tuple()
    X_test_folds=tuple()
    c_test_folds=tuple()
    for i in range(k):
#        X_train_folds[i]=X[:subset_size*(i+1)/2,np.newaxis]
#        X_train_folds=X[:][:n/(k*2)],X[:][:n/k:n/(k-1)],X[:][:n/k*2:n/(k-2)*2]
#        c_train_folds[i]=c[:n/k*2],c[:n/k:n/(k-1)],c[:n*(k-1)/k:len(X)/(k-2)*2]
#        X_test_folds[i]=X[:len(X)/k*2:len(X)/k]
#        c_test_folds[i]=c[:len(X)/k*2:len(X)/k]
        X_train_folds += (X[:,int(n*i/k):int(n*((2*i+1)/2*k))], )
        c_train_folds += (c[int(n*i/k):int(n*((2*i+1)/2*k))], )
        X_test_folds += (X[:,int(n*((2*i+1)/2*k)):int(n*(i+1)/k)], )
        c_test_folds += (c[int(n*((2*i+1)/2*k)):int(n*(i+1)/k)], )
    return X_train_folds, c_train_folds, X_test_folds, c_test_folds
    
X1,c1=extract(X,c)
X_train_folds,c_train_folds,X_test_folds,c_test_folds=kfold_data(X1,c1,5)