import numpy as np
import math

class DecisionTree:
    '''
     This Decision tree class
     ha only been implemented for
     binary classification problems which answers YES/NO type queries

     use <= for cutoff value at each node
    '''
    def __init__(self,max_depth=10,min_sample_leaf=20):
        self.max_depth = max_depth
        self.min_sample_leaf = min_sample_leaf

    def gini_index(self,y_predict,y_actual):
        if(len(y_predict)!=len(y_actual)):
            print("Y_actual and y_predict length doesn't match")
            print(len(y_actual),len(y_predict))
            return 
        n = float(len(y_predict)) 
        c = float(sum(y_predict==y_actual))

        return 1.0-(c/n)**2 - ((n-c)/n)**2


    def get_split_cost(self,x,y):

        min_cost   = 10000000000
        best_cutoff = None
        direction = ""

        for ind,cutoff in enumerate(x):
            
            y_predict = np.array(x<=cutoff) 
            cost = self.gini_index(y_predict,y)
            # print("for cutoff = {}    cost = {}".format(cutoff,cost))
            if(cost<min_cost):
                min_cost = cost
                best_cutoff = cutoff
                direction = "le"
            y_predict = np.array(x>=cutoff) 
            cost = self.gini_index(y_predict,y)
            if(cost<min_cost):
                min_cost = cost
                best_cutoff = cutoff
                direction = "ge"
            
        
        return min_cost,best_cutoff,direction



    def get_best_split(self,x,y):
        
        split_col = -1
        min_cost = 10000000000
        best_cutoff = -1
        direction = ""
        
        for col in range(x.shape[1]):
            cost,cutoff,direction_ = self.get_split_cost(x[:,col],y)
            
            if cost<min_cost:
                min_cost = cost
                split_col = col
                best_cutoff = cutoff
                direction = direction_
        
        return split_col,min_cost,best_cutoff,direction


    def build_tree(self,x,y,depth,prediction):
        print("building node at depth :{}".format(depth))
        if(depth >= self.max_depth or len(x)<=self.min_sample_leaf):    
            return Node(-1,-1,None,None,depth,-1,-1,is_leaf=True,prediction=prediction,samples_in = len(x))

        split_col,min_cost,cutoff,direction = self.get_best_split(x,y)


        x_plus = []
        y_plus = []
        x_minus = []
        y_minus = []
        if direction == "ge":
            for ind,val in enumerate(x):
                if val[split_col]>= cutoff:
                    x_plus.append(val)
                    y_plus.append(y[ind])
                else:
                    x_minus.append(val)
                    y_minus.append(y[ind])
        elif direction == "le":
            for ind,val in enumerate(x):
                if val[split_col]<= cutoff:
                    x_plus.append(val)
                    y_plus.append(y[ind])
                else:
                    x_minus.append(val)
                    y_minus.append(y[ind])

        x_plus = np.array(x_plus)
        y_plus = np.array(y_plus)
        x_minus = np.array(x_minus)
        y_minus = np.array(y_minus)

        n = (len(y))
        c = float(sum(y==True))
        curr_cost = 1 - (c/n)**2 - ((n-c)/n)**2
        if(curr_cost<min_cost):
            node = Node(split_col,cutoff,None,None,depth,min_cost,direction,is_leaf=True,prediction=prediction,samples_in = len(x))
            print("overfitting split found !! \n")
            return node

        left_child = self.build_tree(x_plus,y_plus,depth+1,True)
        right_child = self.build_tree(x_minus,y_minus,depth+1,False)
        node = Node(split_col,cutoff,left_child,right_child,depth,min_cost,direction,is_leaf=False,prediction=prediction,samples_in = len(x))
        
        return node



    def fit(self,X_train,Y_train):
        self.X_train = np.array(X_train)
        self.Y_train = np.array(Y_train)

        self.Root = self.build_tree(self.X_train,self.Y_train,1,True)

    def label(self,x):
        curr_node = self.Root

        while not curr_node.is_leaf:
            col = curr_node.split_col
            cutoff = curr_node.cutoff
            if curr_node.direction == "ge":
                if x[col]>=cutoff:
                    curr_node = curr_node.left_child
                else:
                    curr_node = curr_node.right_child
            else:
                if x[col]<=cutoff:
                    curr_node = curr_node.left_child
                else:
                    curr_node = curr_node.right_child
        
        return curr_node.prediction


    def predict(self,X_test):
        X_test = np.array(X_test)
        y_predict = np.zeros(X_test.shape[0])
        for row in range(X_test.shape[0]):
            y_predict[row] = self.label(X_test[row])

        return y_predict



    def show_tree(self,node):
        node.describe()
        if(node.left_child!=None):
            self.show_tree(node.left_child)
        if(node.right_child!=None):
            self.show_tree(node.right_child)


class Node:
    def __init__(self,split_col,cutoff,left_child,right_child,depth,gini_index,direction,is_leaf=False,prediction=-1,samples_in=-1):
        self.direction = direction
        self.split_col = split_col
        self.cutoff = cutoff
        self.is_leaf = is_leaf
        self.left_child = left_child
        self.right_child = right_child
        self.depth = depth
        self.gini_index = gini_index
        self.prediction = prediction
        self.samples_in = samples_in

    def describe(self):
        print("Samples in : {}".format(self.samples_in))
        print("split_col :{}   cutoff: {}  is_leaf : {}".format(self.split_col,self.cutoff,self.is_leaf))
        print("Depth : {}  Gini_index : {} ".format(self.depth,self.gini_index))
        print("Prediction = {}".format(self.prediction))
        print("\n")
    
