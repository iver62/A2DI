# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 16:19:50 2016

@author: deshayes
"""

import numpy as np
import random as rd
import matplotlib.pyplot as plt

from tp1_ex2.py import ptrain, ptest, Point

dataset = list()

def datagen(n):
    X_train = np.ndarray()
    X_test = np.ndarray()
    c_train = np.array()
    c_test = np.array()
    for i in range(n):
        p = [np.random.random(), np.random.random()]
        dataset.append(p)
    rd.random_shuffle(dataset)
    X_train = dataset[:n*0.1]
    X_test = dataset[n*0.1:]
    for i in range(len(X_train)):
        if (-X_train[i][0]/2 + 0.75) <= -X_train[i][1]:
            c_train.append(1)
        else:
            c_train.append(-1)
    for i in range(len(X_test)):
        if (-X_test[i][0]/2 + 0.75) <= -X_test[i][1]:
            c_test.append(1)
        else:
            c_test.append(-1)
    return X_train, X_test, c_train, c_test
    
def get_test_err(X_train, X_test, c_train, c_test, n):
    
    
X_train, X_test, c_train, c_test = datagen()

err=list()
maxi=25
bin_size=0.5
bins=np.arrange(0.0,np.ceil(maxi+1),bin_size)
bin_centers=bins[:-1]+bin_size/2
crit=1.0
perr=np.ones(bin_centers.shape)
count=0
while crit < 0.001:
    for i in range(20):
        err.append(get_test_err(445))
    perr_old=perr
    (perr,bins_out) = np.histogram(err, bins=bins, normed=True)
    crit = np.max(np.abs(perr-perr_old))
    count += 1
    
plt.figure()
plt.bar(bins[:-1], perr, width=bin_size)
plt.xlabel('test error rate')
plt.title('n=445')

EX1 = np.sum(np.multiply(perr,bin_centers))
print('Erreur de généralisation : ', EX1)