# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 15:37:06 2016

@author: deshayes
"""

import numpy as np
import matplotlib.pyplot as plt

#Q1
taille_h = np.loadtxt("taille_h.txt")
taille_f = np.loadtxt("taille_f.txt")

#Q2
maxi = max(np.max(taille_h), np.max(taille_f))
bins = np.arange(100.0, np.ceil(maxi+1)) - 0.5
bin_centers = bins[:-1] + 0.5
(pX1h, bins_out) = np.histogram(taille_h, bins=bins, normed=True)
plt.figure()
plt.bar(bins[:-1], pX1h, width=1.0)
plt.title('probabilité taille des hommes')

(pX1f, bins_out) = np.histogram(taille_f, bins=bins, normed=True)
plt.figure()
plt.bar(bins[:-1], pX1f, width=1.0)
plt.title('probabilité taille des femmes')

#Q3
ph = 0.505
pf = 0.495
pX1 = ph*pX1h + pf*pX1f
plt.figure()
plt.bar(bin_centers, pX1, width=1.0)
plt.title('probabilité taille hommes et femmes')

#Q4
max_ind = np.argmax(pX1)
print('taille la plus fréquente :', bin_centers[max_ind])
esp = np.sum(np.multiply(pX1, bin_centers))
print('taille moyenne :', esp)
median = np.cumsum(pX1)
i = 0
while median[i] < 0.5:
    i += 1
print('taille médiane :', bin_centers[i-1], ' ', np.median(bin_centers))

#Q5
i = 0
while bin_centers[i] < 180:
    i += 1
ph180 = np.sum(pX1h[i:i+6])
pf180 = np.sum(pX1f[i:i+6])
sum180 = ph180 + pf180
print(ph180/sum180*100,"% d'être un homme,", pf180/sum180*100,"% d'être une femme lorsqu'on mesure entre 1m80 et 1m85")

i = 0
while bin_centers[i] < 160:
    i += 1
ph160 = np.sum(pX1h[i:i+6])
pf160 = np.sum(pX1f[i:i+6])
sum160 = ph160 + pf160
print(ph160/sum160*100,"% d'être un homme,", pf160/sum160*100,"% d'être une femme lorsqu'on mesure entre 1m60 et 1m65")

#Q6
print('Ecart-type hommes :',np.std(taille_h))
print('Ecart-type femmes :',np.std(taille_f))

#Q7