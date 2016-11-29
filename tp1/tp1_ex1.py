import numpy
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy.spatial import distance
data=datasets.load_iris()

# nombre de classes : 3 ('setosa', 'versicolor', 'virginica')
# n = 150 (50 / classe)
# 4 attributs numériques (sepal length, sepal width, petal length, petal width)

print(data.data)
print(data.feature_names)
print(data.target)
print(data.target_names)

dapp = dict()
dapp["array"] = list()
dapp["class"] = list()

dtest = dict()
dtest["array"] = list()
dtest["class"] = list()

for i in range(len(data.data)):
    if (i % 2 == 0):
        dapp.get("array").append(data.data[i])
        dapp.get("class").append(data.target[i])
    else:
        dtest.get("array").append(data.data[i])
        dtest.get("class").append(data.target[i])


def kppv(k, x):
    l = list()
    for i in range(len(dapp["array"])):
        l.append(distance.euclidean(x, dapp["array"][i])) #distance entre x et toutes les données du dataset 
    sorted_list = numpy.argsort(l) #liste croissante des indices des données les plus proches
    classes = list()
    for i in range(len(dapp["array"])):
        classes.append(dtest["class"][sorted_list[i]]) #classe des k plus proches voisins
    return election(classes[:k]) #retourne la classe dominante


def election(int_list):
    return numpy.argmax(numpy.bincount(int_list))


y = list()
for k in range(1, 100):    
    good = 0
    for i in range(len(dtest["array"])):
        p = kppv(k, dtest["array"][i]) #classe prédite
        r = dtest["class"][i] #classe réelle
        if p == r:
            good += 1
        rate = good / len(dtest["array"]) * 100
    y.append(rate)
                
    print('k = ',k, ' ', rate,'%')

plt.plot(y)

