import numpy as np
import matplotlib.pyplot as plt

def datagen(n):
    dataset=list()
    for i in range(n):
        p=[np.random.random() for i in range(2)]
        if (-p[0]/2 + 0.75) <= p[1]:
            p.append(1)
        else:
            p.append(-1)
        dataset.append(p)
    return dataset
    
def split(dataset, d_app_prop, nb_points):
    return dataset[:int(d_app_prop*nb_points)], dataset[int(d_app_prop*nb_points):]
    
def ptrain(d_app):
    theta=np.random.rand(3)
    i=0
    while i < len(d_app):
        x_plus=np.append(d_app[i][:-1],1)
        if sign(np.vdot(theta, x_plus))==d_app[i][2]:
            i+=1
        else:
            theta=theta + d_app[i][2]*x_plus
            i=0
    return theta
    
def ptest(x, theta):
    x_plus=np.append(x[:-1],1)
    return sign(np.vdot(x_plus, theta))

def error(data, theta):
    cpt=0
    for p in data:
        real=p[-1] #classe réelle
        pred=ptest(p,theta) #classe prédite
        if pred!=real:
            cpt+=1
    return cpt/len(data)*100

def sign(x):
    if x >= 0:
        return 1
    return -1
    
dataset=datagen(100)
x_pos=[dataset[i][0] for i in range(100) if dataset[i][2]==1]
y_pos=[dataset[i][1] for i in range(100) if dataset[i][2]==1]
x_neg=[dataset[i][0] for i in range(100) if dataset[i][2]==-1]
y_neg=[dataset[i][1] for i in range(100) if dataset[i][2]==-1]
plt.plot(x_neg,y_neg,'.r',x_pos,y_pos,'.b')
d_app,d_test=split(dataset,0.8,100)
theta=ptrain(d_app)
plt.plot([(-theta[0]*x - theta[2]) / theta[1] for x in range(2)], color="black")

total = 0
for i in range(100):
    total += error(d_test,theta)
print('Erreur de généralisation :',total/100,'%')
