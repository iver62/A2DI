import numpy as np
import matplotlib.pyplot as plt
#from scipy import stats

def datagen(n):
    X=np.random.random((2,n))
    c=[]
    for i in range(n):
        d = np.abs(-0.5*X[0][i] - X[1][i] + 0.75) / (np.sqrt(np.square(0.5) + 1))
#        theta = stats.norm.pdf(d,0,0.1)/stats.norm
        theta = np.exp(-(np.square(d) / 2*np.square(0.05)))
        mistake = np.random.binomial(1,theta/2)
        if (-X[0][i]/2 + 0.75) <= X[1][i]:
            c.append(1)
        else:
            c.append(0)
        if mistake == 1:
            c[i] = 1-c[i]
    c=np.asarray(c)
    return X, c
        
#def kfold_data(X, c, k, n_class):
#    d,n = X.shape
#    X_train_folds=[]
#    c_train_folds=[]
#    X_test_folds=[]
#    c_test_folds=[]
#    myrows = np.array(range(d),dtype=int)
#    for i in range(k):
        
X,c=datagen(300)