# Machine Learning algorithms from scratch
- # Expectation Maximization Algorithm
The Expectation-Maximization (EM) algorithm is a way to find maximum-likelihood estimates for model parameters when your data is incomplete, has missing data points, or has unobserved (hidden) latent variables. It is an iterative way to approximate the maximum likelihood function.

# some outputs of the mixture of gaussian for random inputs
![Alt text](./Figure_1.png "Title")
![Alt text](./Figure_2.png "Title")

# Intuition about EM algorithm from my notes
[LINK TO NOTES](https://smallpdf.com/shared#st=80586d00-97ea-4ece-9b0d-11500a58267a&fn=New+doc+27-Dec-2020+10.30+PM.pdf&ct=1609088550700&tl=share-document&rf=link)
# EM Algorithm
```python
import Expectation_maximization as EM

clf = EM.EM_Model(X,2,iter=10)
clf.fit()
clf.plot_contour()


```
# Implementation (E step and M step)
```python

def E_step(self):
    for i in range(self.N):
        p_t = 0.0
        for j in range (self.k):
            #print("hello")
            p_t = p_t + self.Gaussian(self.X_train[i],self.mean[j],self.covariance_matrix[j])*self.prob[j]
            
        
        for j in range(self.k):
            self.weights[i][j] = self.Gaussian(self.X_train[i],self.mean[j],self.covariance_matrix[j])*self.prob[j]
            self.weights[i][j] = self.weights[i][j]/p_t
def M_step(self):
        sum_weights_col = np.sum(self.weights,axis=0)

        # update the probability values
        self.prob = sum_weights_col/self.N

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
                sig_curr = sig_curr + self.weights[i][j]*np.matmul(x_u.transpose(),x_u)       
            sig_curr = sig_curr/sum_weights_col[j]
            self.covariance_matrix[j] = sig_curr           

```
