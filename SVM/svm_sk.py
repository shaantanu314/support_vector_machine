import numpy as np  # for handling multi-dimensional array operation
import pandas as pd  # for reading data from csv 
from sklearn.preprocessing import MinMaxScaler  # for normalization
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score 
from sklearn.utils import shuffle


class Support_Vector_Machine:
    def __init__(self,visualization = True):
        self.visualization = visualization
        self.weights = []

    def compute_cost(self,W,X,Y):
        distances = 1-Y*(np.dot(X,W))
        distances[distances<0] = 0
        hinge_loss = self.reg_strength * (np.sum(distances) / self.N)
        cost = 1 / 2 * np.dot(W, W) + hinge_loss

        return cost
    
    def calculate_cost_gradient(self,W, X_batch, Y_batch):

        Y_batch = np.array([Y_batch])
        X_batch = np.array([X_batch])
        distance = 1 - (Y_batch * np.dot(X_batch, W))
        dw = np.zeros(len(W))
        for ind, d in enumerate(distance):
            if max(0, d) == 0:
                di = W
            else:
                di = W - (self.reg_strength * Y_batch[ind] * X_batch[ind])
            dw += di

        dw = dw/len(Y_batch)  

        return dw


    def fit(self,features, outputs,learning_rate = 0.00001,reg_strength = 1000,max_epochs = 5000):
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.reg_strength = reg_strength
        self.N = features.shape[0]
        
        weights = np.zeros(features.shape[1])
        nth = 0
        prev_cost = float("inf")
        cost_threshold = 0.01  
        for epoch in range(1, self.max_epochs):
            X, Y = shuffle(features, outputs)
            for ind, x in enumerate(X):

                ascent = self.calculate_cost_gradient(weights, x, Y[ind] )
                weights = weights - (self.learning_rate * ascent)

            if epoch == 2 ** nth or epoch == self.max_epochs - 1:
                cost = self.compute_cost(weights, features, outputs)
                print("Epoch is:{} and Cost is: {}".format(epoch, cost))
                
                if abs(prev_cost - cost) < cost_threshold * prev_cost:
                    self.weights = np.append(self.weights,weights)
                    return weights
                prev_cost = cost
                nth += 1
        self.weights = np.append(self.weights,weights)
        if self.visualization:
            pass

        return weights

    def predict(self,X_test):
        print(self.weights)
        y_predict = np.array([])
        for i in range(X_test.shape[0]):
            yp = np.sign(np.dot(X_test.to_numpy()[i],self.weights))
            y_predict = np.append(y_predict,yp)

        return y_predict