import numpy as np
import matplotlib.pyplot as plt

sigma=0.05

def datagen(n):
    X_train = np.random.rand(int(0.2*n),2)
    X_test = np.random.rand(int(0.8*n),2)
    c_train=np.empty(0, dtype=int)
    c_test=np.empty(0, dtype=int)
    
    for i in range(len(X_train)):
        d = (0.5*X_train[i][0] + X_train[i][1] - 0.75) / (np.sqrt(np.square(0.5) + 1))
        theta = np.exp(-(np.square(d) / 2*np.square(sigma)))
        r = np.random.random()
        if (-X_train[i][0]/2 + 0.75) <= X_train[i][1]:
            if (r > theta/2):
                c_train=np.append(c_train,-1)
            else:
                c_train=np.append(c_train,1)
        else:
            if (r > theta/2):
                c_train=np.append(c_train,1)
            else:
                c_train=np.append(c_train,-1)

    for i in range(len(X_test)):
        d = (0.5*X_test[i][0] + X_test[i][1] - 0.75) / (np.sqrt(np.square(0.5) + 1))
        theta = np.exp(-(np.square(d) / 2*np.square(sigma)))
        if (-X_test[i][0]/2 + 0.75) <= X_test[i][1]:
            if (r > theta/2):
                c_test=np.append(c_test,-1)
            else:
                c_test=np.append(c_test,1)
        else:
            if (r > theta/2):
                c_test=np.append(c_test,1)
            else:
                c_test=np.append(c_test,-1)

    return X_train, X_test, c_train, c_test
    
def aff_dataset(X_train, X_test, c_train, c_test):
    x_pos=list()
    y_pos=list()
    x_neg=list()
    y_neg=list()
    for i in range(len(X_train)):
        if c_train[i]==1:
            x_pos.append(X_train[i][0])
            y_pos.append(X_train[i][1])
        else:
            x_neg.append(X_train[i][0])
            y_neg.append(X_train[i][1])
    for i in range(len(X_test)):
        if c_test[i]==1:
            x_pos.append(X_test[i][0])
            y_pos.append(X_test[i][1])
        else:
            x_neg.append(X_test[i][0])
            y_neg.append(X_test[i][1])
    plt.plot(x_neg,y_neg,'.r',x_pos,y_pos,'.b')

def ptrain_v2(X_train, c_train, X_test, c_test, n_epoch):
    theta=np.random.rand(3)
    best_theta=theta
    best_err=error(X_train,c_train,theta)
    
    for epoch in range(n_epoch):
        for i in range(len(X_train)):
            x_plus=np.append(X_train[i],1)
#            prediction=ptest(X_train[i],theta)
            if sign(np.vdot(theta, x_plus)) != c_train[i]:
                theta = theta + c_train[i]*x_plus
        if error(X_train,c_train,theta) < best_err:
            best_theta=theta
        print(best_theta)
#        print('>epoch=%d, err_train=%.3f, err_test=%.3f' % (epoch,error(X_train,c_train,best_theta),error(X_test,c_test,best_theta)))
    
#    for m in range(max_iterations):
#    i=0
#    m=0
#    while i < len(X_train):
#        for i in range(len(X_train)):
#        x_plus=np.append(X_train[i],1)
#        if sign(np.vdot(theta, x_plus))==c_train[i]:
#            i+=1
#        else:
#            theta=theta + c_train[i]*x_plus
#            m+=1
#            i=0
#            if sign(np.vdot(theta, x_plus)) != c_train[i]:
#                theta = theta + c_train[i]*x_plus
#        print('iteration=',m,'err_train=',error(X_train,c_train,theta),'err_test=',error(X_test,c_test,theta))
#        if error(X_train,c_train,theta) < best_err:
#            best_theta=theta
#        if m==max_iteration:
#            break
    return best_theta

def ptest(x, theta):
    x_plus=np.append(x,1)
    return sign(np.vdot(x_plus, theta))

def error(data, classes, theta):
    cpt=0
    for i in range(len(data)):
        real=classes[i] #classe réelle
        pred=ptest(data[i],theta) #classe prédite
        if pred!=real:
            cpt+=1
    return cpt/len(data)*100
    
def sign(x):
    if x >= 0:
        return 1
    return -1

#def get_test_err(n):
#    X_train, X_test, c_train, c_test = datagen(n)
#    theta = ptrain(X_train, c_train)
#    cpt = 0
#    for i in range(len(X_test)):
#        real = c_test[i]
#        pred = ptest(X_test[i], theta)
#        if pred != real:
#            cpt += 1
#    return cpt / len(X_test) * 100

    
X_train, X_test, c_train, c_test = datagen(300)
aff_dataset(X_train, X_test, c_train, c_test)
theta=ptrain_v2(X_train, c_train, X_test, c_test, 1000)
print(theta)
plt.plot([(-theta[0]*x-theta[2]) / theta[1] for x in range(2)], color="black")