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

X_train_folds, c_train_folds, X_test_folds, c_test_folds = kfold_data(X, c, 3, n_class)