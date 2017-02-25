import scipy as scipy
import numpy as np
from sklearn import cross_validation
#from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

data = scipy.io.loadmat('20news_w100.mat')
X = data["documents"].toarray()
c = data["newsgroups"][0]-1
d,n = X.shape
n_class = len(set(c))


def kfold_data(X, c, k):
    X_train_folds = ()
    c_train_folds = ()
    X_test_folds = ()
    c_test_folds = ()
    kf = cross_validation.KFold(X.shape[1], n_folds=k, shuffle=True)
    kf = list(kf)
    rows = np.arange(X.shape[0])
    for i in range(k):
        X_train_folds += (X[rows[:,np.newaxis],kf[i][0]], )
        X_test_folds += (X[rows[:,np.newaxis],kf[i][1]], )
        c_train_folds += (np.array([c[j] for j in kf[i][0]]), )
        c_test_folds += (np.array([c[j] for j in kf[i][1]]), )
    return X_train_folds, c_train_folds, X_test_folds, c_test_folds
    
#def kfold_data(X, c, k, n_class):
#    X_train_folds = ()
#    c_train_folds = ()
#    X_test_folds = ()
#    c_test_folds = ()
#    for i in range(k):
#        X_train, X_test, c_train, c_test = train_test_split(X.T, c, test_size=0.4, random_state=np.random.randint(100))
#        X_train_folds += (X_train.T,)
#        c_train_folds += (c_train,)
#        X_test_folds += (X_test.T,)
#        c_test_folds += (c_test,)
#    return X_train_folds, c_train_folds, X_test_folds, c_test_folds
    
def aff_dataset(X_train, c_train, nb):
    zeros = np.zeros((100, nb), dtype=int) #les nb premiers exemples de la classe 0
    ones = np.zeros((100, nb), dtype=int) #les nb premiers exemples de la classe 1
    cpt0 = 0
    cpt1 = 0
    for i in range(X_train.shape[1]):
        if c_train[i] == 0 and cpt0 < nb:
            zeros[:, cpt0] = X_train[:, i]
            cpt0 += 1
        elif c_train[i] == 1 and cpt1 < nb:
            ones[:, cpt1] = X_train[:, i]
            cpt1 += 1
        if cpt0 == nb and cpt1 == nb:
            break
    plt.imshow(np.vstack((zeros, ones)), cmap='Greys_r')
    
def distribution(c_train):
    return np.bincount(c_train)/len(c_train)
    
def cond_prob(X_train, c_train):
    cond = np.zeros((d, n_class))
    for i in range(n_class):
        tmp = np.array([X_train[:,j] for j in range(len(c_train)) if c_train[j] == i]) #tous les mots de la classe i
        for j in range(d):
            p = np.bincount(tmp[j])[1]/tmp.shape[1]
            cond[j][i] = p
    return cond


X_train_folds, c_train_folds, X_test_folds, c_test_folds = kfold_data(X, c, 3)
aff_dataset(X_train_folds[0], c_train_folds[0], 200)

for i in range(3):
    print('pli n°',i+1,':',distribution(c_train_folds[i]))

for i in range(3):
    print('pli n°',i+1,':\n',cond_prob(X_train_folds[i], c_train_folds[i]))
