import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import datasets
from scipy.spatial import distance

data=datasets.load_iris()
n=len(data.data)
nb_class=len(data.target_names)
d=len(data.feature_names)

print('nombre d\'exemples :', n)
print('nombre de classes :', nb_class)
print('nombre d\'attributs :', d)
print('attributs :',' '.join(data.feature_names))

def split(X, c, p):
    n_test = int(n * (1-p))
    n_train = n - n_test
    X_train = np.zeros((n_train, d))
    c_train = np.zeros(n_train)
    X_test = np.zeros((n_test, d))
    c_test = np.zeros(n_test)

#    X_train = np.zeros((int(n*p), d))
#    c_train = np.zeros(int(n*p))
#    X_test = np.zeros((int(n*(1-p)), d))
#    c_test = np.zeros(int(n*(1-p)))
    ix_train = 0
    ix_test = 0
    for i in range(n):
        if (i%(int(1/p)) == 0):
#            X_train[int(i*p)] = X[i]
#            c_train[int(i*p)] = c[i]
            X_train[ix_train] = X[i]
            c_train[ix_train] = c[i]
            ix_train += 1
        else:
            X_test[ix_test] = X[i]
            c_test[ix_test] = c[i]
            ix_test += 1
    return X_train, c_train, X_test, c_test

def kppv(x, X_train, c_train, k):
    distances = []
    classes = []
    for i in range(len(X_train)):
        distances.append(distance.euclidean(x, X_train[i])) #distance entre x et toutes les données de l'ensemble d'apprentissage 
    sorted_list = np.argsort(distances) #liste croissante des indices des données les plus proches
    for i in range(len(X_train)):
        classes.append(c_train[sorted_list[i]]) #classes des k plus proches voisins
    return np.argmax(np.bincount(classes[:k])) #retourne la classe dominante

def get_test_err(X_train, c_train, X_test, c_test, k):
    good = 0
    for i in range(X_test.shape[0]):
        p = kppv(X_test[i], X_train, c_train, k) #classe prédite
        r = c_test[i] #classe réelle
        if p == r:
            good += 1
    rate = good / len(X_test) * 100 #taux de bonne classification
    return rate

X = data.data
c = data.target
X_train, c_train, X_test, c_test = split(X, c, 0.30)
rates = []
for k in range(9, 9):
    rate = get_test_err(X_train, c_train, X_test, c_test, k)
    rates.append(rate)                
    print('k =', k, rate,'%')
plt.plot(rates)

#times = []
#props = np.arange(0.2, 1, 0.05)
#for p in props:
#    print(p)
#    rates = []
#    X_train, c_train, X_test, c_test = split(X, c, p)
#    times_list = []
#    for k in range(1, 101):
#        t = time.time()
#        rate = get_test_err(X_train, c_train, X_test, c_test, k)
#        times_list.append(time.time() - t)
#        if p == 0.5:
#            rates.append(rate)
#            print('k =', k, rate,'%')
#    if p == 0.5:
#        plt.figure()
#        plt.plot(rates)
#    times.append(np.mean(times_list))
#plt.figure()
#plt.plot(props, times)