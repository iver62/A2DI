# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 15:51:10 2017

@author: deshayes
"""

import numpy as np

class Point:
    
    def __init__(self, x1, x2):
        self.x1 = x1
        self.x2 = x2
        self.cl = 0
        self.d = 0
        
    def setClass(self):
        if (-self.x1/2 + 0.75) <= self.x2:
            self.cl = 1
        else:
            self.cl = -1
            
    def setDistance(self):
        return abs(0.5*self.x1 + self.x2 - 0.75) / (np.sqrt(np.square(0.5) + 1))
    
    def getDistance(self):
        return self.d
        
    def getClass(self):
        return self.cl
        
    def theta(self, sigma):
        return np.exp(-(np.square(self.d) / 2 * np.square(sigma)))
        
    def toString(self):
        return self.x1,self.x2,self.cl