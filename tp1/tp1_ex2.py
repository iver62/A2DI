# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 10:59:07 2016

@author: Pierrick
"""

import numpy
import random
import matplotlib.pyplot as plt

nbPoints = 100

dataset = list()
dapp = list()
dtest = list()

class Point:
    
    def __init__(self, x1, x2):
        self.x1 = x1
        self.x2 = x2
        self.cl = 0
        
    def setClass(self):
        if (-self.x1/2 + 0.75) <= self.x2:
            self.cl = 1
        else:
            self.cl = -1
            
    def getClass(self):
        return self.cl
        
    def toString(self):
        return self.x1,self.x2,self.cl


def datagen():
    for i in range(nbPoints):
        p = Point(numpy.random.random(), numpy.random.random())
        p.setClass()
        dataset.append(p)
#        print(p.toString())
    

def sign(x):
    if x >= 0:
        return 1
    return -1

 
def repartition():
    for i in range(int(0.8*100)):
        dapp.append(dataset.pop(random.randint(0, len(dataset)-1)))
    for i in range(len(dataset)):
        dtest.append(dataset[i])
   

def ptrain():
    theta = numpy.array([numpy.random.random(),numpy.random.random(),numpy.random.random()])
    i = 0
    while i < len(dapp):
        xplus = numpy.array([dapp[i].x1, dapp[i].x2, 1])
        if sign(numpy.vdot(theta, xplus)) == dapp[i].cl:
            i += 1
        else:
            theta = theta + dapp[i].cl*xplus
            i = 0
    return theta


def ptest(x):
    xplus = numpy.array([x.x1, x.x2, 1])
    return sign(numpy.vdot(xplus, theta))

datagen()
    
xNeg = list()
yNeg = list()
xPos = list()
yPos = list()

for i in range(nbPoints):
    if dataset[i].cl == -1:
        xNeg.append(dataset[i].x1)
        yNeg.append(dataset[i].x2)
    else:
        xPos.append(dataset[i].x1)
        yPos.append(dataset[i].x2)

plt.plot(xNeg,yNeg,"or")
plt.plot(xPos,yPos,"ob")


repartition()

    
theta = ptrain()

w1 = theta[0]
w2 = theta[1]
b = theta[2]
print(w1, w2, b)
    
plt.plot([(-w1*x - b) / w2 for x in range(2)], color="black")

total = 0
for i in range(100):
    cpt = 0
    for p in dtest:
        real = p.getClass()
        pred = ptest(p)
        if pred == real:
            cpt += 1
#        print("classe reelle : ", real, " ,classe prédite : ", pred)
    total += cpt/len(dtest) * 100
    
print(total/100, "% de bonne classification")