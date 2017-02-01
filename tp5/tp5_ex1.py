import scipy as scipy
import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

data = scipy.io.loadmat('20news_w100.mat')
X = data["documents"].toarray()
c = data["newsgroups"][0]-1
d,n = X.shape
n_class = len(set(c))
print(n,'exemples')
print('dimension =',d)
print(n_class,'classes')

    
def kfold_data(X, c, k, n_class):
    X_train_folds = ()
    c_train_folds = ()
    X_test_folds = ()
    c_test_folds = ()
    for i in range(k):
        X_train, X_test, c_train, c_test = train_test_split(X.T, c, test_size=0.4, random_state=np.random.randint(100))
        X_train_folds += (X_train.T,)
        c_train_folds += (c_train,)
        X_test_folds += (X_test.T,)
        c_test_folds += (c_test,)
    return X_train_folds, c_train_folds, X_test_folds, c_test_folds
    
def aff_dataset(X_train, c_train, nb):
    zeros = np.zeros((100, nb), dtype=int)
    ones = np.zeros((100, nb), dtype=int)
    cpt0 = 0
    cpt1 = 0
    for i in range(X_train.shape[1]):
        if c_train[i] == 0 and cpt0 < 100:
            zeros[:,cpt0] = X_train[:,i]
            cpt0 += 1
        elif c_train[i] == 1 and cpt1 < 100:
            ones[:,cpt1] = X_train[:,i]
            cpt1 += 1
        if cpt0 == 100 and cpt1 == 100:
            break
    plt.imshow(np.vstack((zeros,ones)), cmap='Greys_r')
    
def proba(X_train, c_train):
    return np.bincount(c_train)/len(c_train)
    
def cond_prob(X_train, c_train):
    cond = np.zeros((d,n_class))
    for i in range(n_class):
        tmp = np.array([X_train[:,j] for j in range(len(c_train)) if c_train[j] == i])
        for j in range(d):
            p = np.bincount(tmp[j])[1]/tmp.shape[1]
            cond[j][i] = p
    return cond


X_train_folds, c_train_folds, X_test_folds, c_test_folds = kfold_data(X, c, 3, n_class)
aff_dataset(X_train_folds[0], c_train_folds[0], 100)

for i in range(3):
    print('pli n°',i+1,':',proba(X_train_folds[i], c_train_folds[i]))

for i in range(3):
    print('pli n°',i+1,':\n',cond_prob(X_train_folds[i], c_train_folds[i]))
