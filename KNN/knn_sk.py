import numpy as np
import math
import statistics

class KNN:
    def __init__(self,X_train,Y_train,k,regression=False,classification=False):
        self.regression = regression
        self.classification = classification
        self.X_train = np.array(X_train)
        self.Y_train = np.array(Y_train)
        self.k = k

    def knn(self,X_test):
        X_test = np.array(X_test)
        y_test = np.zeros(X_test.shape[0])

        for ind,x in enumerate(X_test):
            y_test[ind] =  int(self.predict(x))
        
        return y_test

    def predict(self,x):
        nearest_neighbours_dist = []
        for index,x_train in enumerate(self.X_train):
            dist = self.euclidean_dist(x,x_train)
            nearest_neighbours_dist.append((dist,index))
        sorted_nearest_neighbours = sorted(nearest_neighbours_dist)                      
        if self.classification:
            tie = True
            k = self.k
            while tie:
                k_nearest_neighbours = sorted_nearest_neighbours[:k]
                k_nearest_neighbours_labels = [self.Y_train[i[1]] for i in k_nearest_neighbours]
                
                # try:
                prediction  = int(statistics.mode(k_nearest_neighbours_labels))
                # return prediction
                # except:
                #     k = k-1
                #     print("Tie in classification found : Trying for k = %s " %(k))             
                # 
                tie = False
            return prediction     

        elif self.regression:
            #to be coded
            pass



    def euclidean_dist(self,x,x_train):
        squared_sum = 0
        for ind,feature in enumerate(x):
            squared_sum = squared_sum + math.pow(x[ind]-x_train[ind],2)
        
        return math.sqrt(squared_sum)



        
