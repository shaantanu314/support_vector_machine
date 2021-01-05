from scipy.io import wavfile
from sklearn.decomposition import FastICA
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import math 

n_samples = 2000
time = np.linspace(0,8,n_samples)
alpha = 0.1
h = np.zeros(4000)
def center(X):
    X = np.array(X)
    mean = X.mean(axis=0, keepdims=True)
    return X- mean

def whitening(X):
    # cov = np.cov(X)
    cov = np.matmul(X,X.T)
    d, E = np.linalg.eigh(cov)

    D = np.diag(d)
    print(d)
    D_inv = np.sqrt(np.linalg.inv(D))
    X_whiten = np.dot(E, np.dot(D_inv, np.dot(E.T, X)))
    return X_whiten

# def g(x):
#     return 1/(1+np.exp(-x))
# def g_der(x):
#     return (1 - g(x)) * g(x)

def g(x):
    return np.tanh(x)

def g_der(x):
    return 1 - g(x) * g(x)

def log_likelihood(X,W):
    value = 0.0
    det = np.linalg.det(W)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            value = value + math.log(g_der(np.dot(W[j],X[i].T)))
    
    return value

def new_w(X,W,iter):
    A = np.zeros((X.shape[0],1))
    new_W = W
    N = X.shape[1]
    # for j in range(X.shape[1]):
    #     for i in range(X.shape[0]):
    #         A[i] = 1 - 2/N*g(np.dot(W[i],X.T[j].T))
        
    #     new_W = new_W + alpha*((A*X.T[j]) + np.linalg.inv(new_W.T))
    h[iter] =  1/N*sum(sum(g(np.matmul(X.T,W)))) +0.5*np.log(np.linalg.det(W))
    new_W = W + alpha*(np.linalg.inv(W.T) -2/N*np.matmul(X,g(np.matmul(X.T,W)))  )
    return new_W


s1 = np.sin(2 * time)  
s2 = np.sign(np.sin(3 * time))  
s3 = signal.sawtooth(2 * np.pi * time)  

X = np.c_[s1,s2,s3]

# A = np.array(([[1, 0.5], [0.5, 2]]))

# a = np.random.laplace(size=(1,500))
# b = np.random.laplace(size=(1,500))
# X = np.c_[a.T , b.T]
# X = X.T
# print X.shape
# X = np.dot(A,X)
# plt.figure()
# plt.scatter(X[0],X[1])
# plt.show()

# plt.figure()
# for i in range(X.shape[1]):
#     plt.plot(X.T[i])
# plt.show()


A = np.array(([[0.8, 0.1, 0.1], [0.1, 0.6, 0.3], [0.15, 0.05, 0.9]]))



X = np.dot(X,A.T)
X = X.T
plt.figure()
for i in range(X.shape[0]):
    plt.plot(X[i])
plt.show()

print(X.shape)
X = center(X)
# X = whitening(X)


# plt.figure()
# for i in range(X.shape[0]):
#     plt.plot(X[i])
# plt.show()

W = np.random.rand(3,3)

for i in range(4000):
    W = new_w(X,W,i)
print W
# print log_likelihood(X,W)

plt.figure()
plt.plot(h)
plt.show()


print "done"
print X.shape
print W.shape
S = np.dot(W,X)


plt.figure()
for i in range(S.shape[0]):
    plt.plot(S[i])
plt.show()



# # USING FastICA from sklearn
# ica = FastICA(n_components=3)
# S = ica.fit_transform(X.T)

# print S.shape
# plt.figure()
# for i in range(S.shape[1]):
#     plt.plot(S.T[i])
# plt.show()