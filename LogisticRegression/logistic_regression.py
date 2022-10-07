import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Leggiamo i dati e ricaviamo il nostro numpy array

dataset = pd.read_csv('Bank Customer Churn Prediction.csv', ).drop(['customer_id', 'country', 'gender'], axis = 1).values
label = dataset[:,-1:]
data = dataset[:,:-1]
# creiamo il vettore dei pesi ricordando di aggiungere il bias
theta = np.zeros((np.shape(data)[1] + 1,1))

# normalizziamo i dati altrimenti quello che succede è che l'esponenziale diventa enorme e python divide tutto per zero
data_norm = data / data.max(axis = 0)
data_norm = np.c_[np.ones((np.shape(data)[0],1)),data_norm]


z = np.dot(data_norm, theta)
def sigmoid ( Z ) :
    sig = 1/(1 + np.exp(-Z))

    return sig

def gradient( X, Z, Y ):
    return np.dot(X.T, sigmoid(Z) - Y)/len(Y)

def cost(Z,Y) :
    cost1 = np.dot(Y.T, np.log(sigmoid(Z)))
    cost2 = np.dot((1-Y).T,np.log(1-sigmoid(Z)))

    return -((cost1 + cost2))/len(Y)


epoch = 10000
lr = 0.01
cost_list = np.zeros(epoch,)

for i in range(epoch) :
    theta = theta - lr*gradient(data_norm,z,label)
    z = np.dot(data_norm,theta)
    cost_list[i] = cost(z,label)


def prediction( X, weights ) :
    z = np.dot(X, weights)
    list = []
    for j in  sigmoid(z) :
        if j >= 0.5:
            list.append(1)
        else :
            list.append(0)
    return list

def f1_score(y,y_hat):
    tp,tn,fp,fn = 0,0,0,0
    for j in range(len(y)):
        if y[j] == 1 and y_hat[i] == 1:
            tp += 1
        elif y[j] == 1 and y_hat[i] == 0:
            fn += 1
        elif y[j] == 0 and y_hat[i] == 1:
            fp += 1
        elif y[j] == 0 and y_hat[i] == 0:
            tn += 1
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score = 2*precision*recall/(precision+recall)
    return f1_score


y_pred = prediction(data_norm, theta)
f1_score = f1_score(label,y_pred)

print(f1_score)
