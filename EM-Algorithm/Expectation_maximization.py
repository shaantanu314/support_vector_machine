import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import interpolate

class EM_Model:

    def __init__(self,X_train,k,iter=10):
        self.iter = iter
        self.k = k
        self.X_train = np.array(X_train)
        self.N = self.X_train.shape[0] # number of examples
        self.d = self.X_train.shape[1] # dimension

        self.mean              = np.zeros([self.k,self.d])
        self.covariance_matrix = np.zeros([self.k,self.d,self.d])
        self.prob              = np.zeros([1,self.k])

    def M_step(self):
        sum_weights_col = np.sum(self.weights,axis=0)

        # update the probability values
        self.prob = sum_weights_col/self.N
        # print("pre prob")
        # print(self.prob)
        # print(sum_weights_col)
        # print("\n")

        # update mean of distribution    
        for j in range(self.k):
            a = np.matmul(self.weights[:,j],self.X_train)
            a = a/sum_weights_col[j]
            self.mean[j] = a
        
        # update covariance matrix
        for j in range(self.k):
            sig_curr = np.zeros([self.d,self.d])
            for i in range(self.N):
                x_u = self.X_train[i] - self.mean[j]
                x_u = np.matrix(x_u)
                # print(x_u)
                # print(x_u.transpose())
                # print("\n")
                sig_curr = sig_curr + self.weights[i][j]*np.matmul(x_u.transpose(),x_u)       
            sig_curr = sig_curr/sum_weights_col[j]
            print(sig_curr)
            # print("\n")
            self.covariance_matrix[j] = sig_curr





    def E_step(self):
        for i in range(self.N):
            p_t = 0.0
            for j in range (self.k):
                #print("hello")
                p_t = p_t + self.Gaussian(self.X_train[i],self.mean[j],self.covariance_matrix[j])*self.prob[j]
                
            
            for j in range(self.k):
                self.weights[i][j] = self.Gaussian(self.X_train[i],self.mean[j],self.covariance_matrix[j])*self.prob[j]
                self.weights[i][j] = self.weights[i][j]/p_t
            

    def Gaussian(self,x,u,sig):
        x_u = x - u
        #print(sig)
        sig_inv = np.linalg.inv(sig)
        a = np.power(2*np.pi,self.d)*np.linalg.det(sig)
        expo = np.exp(-0.5*np.matmul(np.matmul(x_u,sig_inv),x_u.T))
        a = np.sqrt(a)
        return expo*a

    def fit(self):
        self.init_weights()
        # This initiallized the weights for each training example such that sum of all probabilities for 
        # each of the k possible cluster add up to 1
        self.M_step()
        for i in range(self.iter):
            # print("prob:")
            # print(self.prob)

            self.E_step()
            self.M_step()

        print("mean:")
        print(self.mean)
        print("prob:")
        print(self.prob)

            
        

    def init_weights(self):
        self.weights = np.zeros([self.N,self.k])
        for j in range(self.N):
            ind = np.random.randint(0,self.k)
            self.weights[j][ind]=1.0
            # r = [np.random.random() for i in range(self.k)]
            # s = sum(r)
            # r = [ i/s for i in r ]
            # self.weights[j] = r
        

    def plot_contour(self):
        coordinates = [[x , y] for x in range(1,200)  for y in range(1,200)]
        coordinates = np.array(coordinates)
        coordinates = coordinates/2
        # print(coordinates)
        z = np.zeros(coordinates.shape[0])
        for j,test in enumerate(coordinates):
            for i in range(self.k):
                #print("ok")
                z[j] = z[j] + self.Gaussian(test,self.mean[i],self.covariance_matrix[i])
        z = z.reshape(199,199)
        # X = np.linspace(1,100,199);
        # Y = np.linspace(1,100,199);
        X, Y = np.mgrid[0:1:199j, 0:1:199j]
   
        print self.covariance_matrix
        plt.figure()
        plt.contourf(X,Y, z)
        plt.show()
        # print X
        # plt.figure()
        # plt.contourf(X,Y,z);
        # plt.show()


