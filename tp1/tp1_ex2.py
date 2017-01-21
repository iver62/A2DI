import numpy as np
import matplotlib.pyplot as plt

def datagen(n):
    X=np.random.random((2,n))
    classes=np.empty(0,dtype=int)
    for i in range(n):
        if (-X[0][i]/2 + 0.75) <= X[1][i]:
            classes=np.append(classes,1)
        else:
            classes=np.append(classes,-1)
    return np.vstack((X,classes))
    
def split(D):
    return D[:,:int(0.8*len(D[0]))], D[:,int(0.8*len(D[0])):]
    
def ptrain(D_app):
    theta=np.random.random((3,1))
    i=0
    while i < len(D_app[0]):
        x_plus=np.concatenate((D_app[:-1,[i]],[[1]]),axis=0)
        if sign(np.vdot(theta, x_plus))==D_app[2][i]:
            i+=1
        else:
            theta=theta + D_app[2][i]*x_plus
            i=0
    return theta
    
def ptest(x, theta):
    x_plus=np.concatenate((x,[[1]]),axis=0)
    return sign(np.vdot(x_plus, theta))

def get_test_err(n):
    D=datagen(n)
    D_app,D_test=split(D)
    theta=ptrain(D_app)
    aff_dataset(D,theta)
    cpt=0
    for i in range(len(D_test[0])):
        real=D_test[-1,i] #classe réelle
        pred=ptest(D_test[:-1,[i]],theta) #classe prédite
        if pred!=real:
            cpt+=1
    return cpt/len(D_test[0])*100

def sign(x):
    if x >= 0:
        return 1
    return -1

def aff_dataset(D, theta):
    x_pos=[D[0][i] for i in range(100) if D[2][i]==1]
    y_pos=[D[1][i] for i in range(100) if D[2][i]==1]
    x_neg=[D[0][i] for i in range(100) if D[2][i]==-1]
    y_neg=[D[1][i] for i in range(100) if D[2][i]==-1]
    plt.figure()
    plt.plot(x_neg,y_neg,'.r',x_pos,y_pos,'.b')
    D_app,D_test=split(D)
    theta=ptrain(D_app)
    plt.plot([(-theta[0]*x - theta[2]) / theta[1] for x in range(2)], 'k')

err=list()
for i in range(20):
    print('i=',i)
    err.append(get_test_err(100))
print('Erreur de généralisation :',np.mean(err),'%')