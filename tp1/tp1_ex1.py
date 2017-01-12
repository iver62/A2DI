import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy.spatial import distance

data=datasets.load_iris()

n=len(data.data)
nb_class=len(data.target_names)
d=len(data.feature_names)

print('nombre d\'exemples :',n)
print('nombre de classes :',nb_class)
print('nombre d\'attributs :',d)
print('attributs :',' '.join(data.feature_names))

def split(X, C):
    d_app=list()
    c_app=list()
    d_test=list()
    c_test=list()
    for i in range(len(X)):
        if (i % 2 == 0):
            d_app.append(X[i])
            c_app.append(C[i])
        else:
            d_test.append(X[i])
            c_test.append(C[i])
    return d_app, c_app, d_test, c_test

def kppv(x, d_app, c_app, k):
    distances=list()
    classes=list()
    for i in range(len(d_app)):
        distances.append(distance.euclidean(x, d_app[i])) #distance entre x et toutes les données de l'ensemble d'apprentissage 
    sorted_list=np.argsort(distances) #liste croissante des indices des données les plus proches
    for i in range(len(d_app)):
        classes.append(c_app[sorted_list[i]]) #classes des k plus proches voisins
    return np.argmax(np.bincount(classes[:k])) #retourne la classe dominante
    
d_app, c_app, d_test, c_test = split(data.data, data.target)

rates=list()
for k in range(1, 100):    
    good = 0
    for i in range(len(d_test)):
        p = kppv(d_test[i], d_app, c_app, k) #classe prédite
        r = c_test[i] #classe réelle
        if p == r:
            good += 1
        rate = good / len(d_test) * 100
    rates.append(rate)
                
    print('k = ',k, ' ', rate,'%')

plt.plot(rates)