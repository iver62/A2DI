# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 15:50:47 2017

@author: deshayes
"""

import numpy as np
import scipy as scipy

data=scipy.io.loadmat('20news_w100.mat')
X=data["documents"].toarray()
c=data["newsgroups"][0]-1
d,n=X.shape
n_class=len(set(c))
print(n,'exemples')
print('dimension =',d)
print(n_class,'classes')

def kfold_data(X, c, k, n_class):
    subset_size = len(X)/k
    X_train_folds=np.ndarray(2)
    c_train_folds=np.ndarray(1)
    X_test_folds=np.ndarray(2)
    c_test_folds=np.ndarray(1)
    for i in range(k):
        X_train_folds[i]=X[:subset_size*(i+1)/2,np.newaxis]
        c_train_folds[i]=c[:subset_size*i:subset_size*(i+1)/2]
        X_test_folds[i]=X[:subset_size*(i+1)/2:subset_size*(i+1),np.newaxis]
        c_test_folds[i]=c[:subset_size*(i+1)/2:subset_size*(i+1)]
    return X_train_folds, c_train_folds, X_test_folds, c_test_folds

X_train_folds, c_train_folds, X_test_folds, c_test_folds = kfold_data(X, c, 3, n_class)