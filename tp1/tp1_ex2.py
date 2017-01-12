# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 10:59:07 2016

@author: Pierrick
"""

import numpy as np
import matplotlib.pyplot as plt

def datagen(nb_points):
    dataset=list()
    for i in range(nb_points):
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
    theta=np.array([np.random.random() for i in range(3)])
    i=0
    while i < len(d_app):
        xplus=np.array([d_app[i][0], d_app[i][1], 1])
        if sign(np.vdot(theta, xplus)) == d_app[i][2]:
            i+=1
        else:
            theta=theta + d_app[i][2]*xplus
            i=0
    return theta
    
def sign(x):
    if x >= 0:
        return 1
    return -1
    
def ptest(x, theta):
    xplus=np.array([x[0], x[2], 1])
    return sign(np.vdot(xplus, theta))

dataset=datagen(100)
x_pos=[dataset[i][0] for i in range(100) if dataset[i][2]==1]
y_pos=[dataset[i][1] for i in range(100) if dataset[i][2]==1]
x_neg=[dataset[i][0] for i in range(100) if dataset[i][2]==-1]
y_neg=[dataset[i][1] for i in range(100) if dataset[i][2]==-1]
plt.plot(x_neg,y_neg,'or')
plt.plot(x_pos,y_pos,'ob')
d_app,d_test=split(dataset,0.8,100)
theta=ptrain(d_app)
plt.plot([(-theta[0]*x - theta[2]) / theta[1] for x in range(2)], color="black")

total = 0
for i in range(100):
    cpt=0
    for p in d_test:
        real=p[2]
        pred=ptest(p,theta)
        if pred==real:
            cpt+=1
    total += cpt/len(d_test)*100 
print(total/100, "% de bonne classification")
