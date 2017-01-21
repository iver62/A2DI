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

def aff_dataset(X_train, X_test, c_train, c_test):
    x_pos=list()
    y_pos=list()
    x_neg=list()
    y_neg=list()
    for i in range(len(X_train[0])):
        if c_train[i]==1:
            x_pos.append(X_train[0][i])
            y_pos.append(X_train[1][i])
        else:
            x_neg.append(X_train[0][i])
            y_neg.append(X_train[1][i])
    for i in range(len(X_test[0])):
        if c_test[i]==1:
            x_pos.append(X_test[0][i])
            y_pos.append(X_test[1][i])
        else:
            x_neg.append(X_test[0][i])
            y_neg.append(X_test[1][i])
    plt.plot(x_neg,y_neg,'.r',x_pos,y_pos,'.b')
      
#def kfold_data(X, c, k, n_class):
#    d,n = X.shape
#    X_train_folds=[]
#    c_train_folds=[]
#    X_test_folds=[]
#    c_test_folds=[]
#    myrows = np.array(range(d),dtype=int)
#    for i in range(k):
        
X,c=datagen(300)
plt.plot(X[0],X[1],'.')
aff_dataset(X[:,:int(0.2*len(X))],X[:,int(0.2*len(X)):],c[:int(0.2*len(X))],c[int(0.2*len(X)):])