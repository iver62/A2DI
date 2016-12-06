# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 15:37:06 2016

@author: deshayes
"""

import numpy as np
import matplotlib.pyplot as plt

taille_h = np.loadtxt("taille_h.txt")
taille_f = np.loadtxt("taille_f.txt")

plt.hist(taille_h, bins=len(taille_h), color='blue')
plt.title('taille des hommes')

plt.hist(taille_f, bins=len(taille_f), color='pink')
plt.title('taille des femmes')