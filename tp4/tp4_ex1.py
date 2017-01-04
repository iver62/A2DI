# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 16:24:21 2017

@author: deshayes
"""

import numpy as np

from sklearn import datasets
data=datasets.load_digits()
X=data.data
X=X.T
c=data.target
print(X)

X1=np.ndarray(2)
c1=np.array(1)

for i in range(len(c)):
    if c[i] == 5 or c[i] == 6:
        np.append(X1,X[i],0)
        np.append(c1,c[i])