# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 15:47:25 2017

@author: deshayes
"""

import numpy as np
import random as rd
import matplotlib.pyplot as plt
#import math as math
#import point

sigma = 0.05

dataset = list()

def datagen(n):
    X_train = np.ndarray(2)
    X_test = np.ndarray(2)
    c_train = list()
    c_test = list()
    for i in range(n):
        p = [np.random.random(), np.random.random()]
        dataset.append(p)
    rd.shuffle(dataset)
    X_train = dataset[:int(n*0.2)]
    X_test = dataset[-int(n*0.8):]
    
    for i in range(len(X_train)):
        d = (0.5*X_train[i][0] + X_train[i][1] - 0.75) / (np.sqrt(np.square(0.5) + 1))
        theta = np.exp(-(np.square(d) / 2 * np.square(sigma)))
        r = np.random.random()
        if (-X_train[i][0]/2 + 0.75) <= X_train[i][1]:
            if (r < theta/2):
                c_train.append(-1)
            else:
                c_train.append(1)
        else:
            if (r < theta/2):
                c_train.append(1)
            else:
                c_train.append(-1)
                
    for i in range(len(X_test)):
        d = (0.5*X_test[i][0] + X_test[i][1] - 0.75) / (np.sqrt(np.square(0.5) + 1))
        theta = np.exp(-(np.square(d) / 2*np.square(sigma)))
        r = np.random.random()
        if (-X_test[i][0]/2 + 0.75) <= X_test[i][1]:
            if (r < theta/2):
                c_test.append(-1)
            else:
                c_test.append(1)
        else:
            if (r < theta/2):
                c_test.append(1)
            else:
                c_test.append(-1)
                
    return X_train, X_test, c_train, c_test

X_train, X_test, c_train, c_test = datagen(300)

xNeg = list()
yNeg = list()
xPos = list()
yPos = list()

for i in range(len(X_train)):
    if c_train[i] == -1:
        xNeg.append(X_train[i][0])
        yNeg.append(X_train[i][1])
    else:
        xPos.append(X_train[i][0])
        yPos.append(X_train[i][1])
        
for i in range(len(X_test)):
    if c_test[i] == -1:
        xNeg.append(X_test[i][0])
        yNeg.append(X_test[i][1])
    else:
        xPos.append(X_test[i][0])
        yPos.append(X_test[i][1])        

plt.plot(xNeg,yNeg,"or")
plt.plot(xPos,yPos,"ob")