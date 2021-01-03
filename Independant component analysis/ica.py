from scipy.io import wavfile
from sklearn.decomposition import FastICA
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import math 

n_samples = 2000
time = np.linspace(0,8,n_samples)
alpha = 0.1

def center(X):
    X = np.array(X)
    mean = X.mean(axis=0, keepdims=True)
    return X- mean

def whitening(X):
    cov = np.cov(X)
    d, E = np.linalg.eigh(cov)
    D = np.diag(d)
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

def new_w(X,W):
    A = np.zeros((X.shape[1],1))
    new_W = W
    for j in range(X.shape[0]):
        for i in range(X.shape[1]):
            A[i] = 1 - 2*g(np.dot(W[i],X[j].T))
        new_W = new_W + alpha*((A*X[j]) + np.linalg.inv(new_W.T))

    return new_W


s1 = np.sin(2 * time)  
s2 = np.sign(np.sin(3 * time))  
s3 = signal.sawtooth(2 * np.pi * time)  

X = np.c_[s1,s2,s3]

X = np.array(X)
print X.shape
plt.figure()
for i in range(X.shape[1]):
    plt.plot(X.T[i])
plt.show()


A = np.array(([[1, 5, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]]))

# X = center(X)


X = np.dot(X,A.T)


# X = whitening(X)
print(X.shape)
plt.figure()
for i in range(X.shape[1]):
    plt.plot(X.T[i])
plt.show()

W = np.random.rand(3,3)

for i in range(200):
    print(i)
    W = new_w(X,W)
print W
# print log_likelihood(X,W)

S = np.dot(X,W)
S = S.T

plt.figure()
for i in range(S.shape[0]):
    plt.plot(S[i])
plt.show()



# USING FastICA from sklearn
ica = FastICA(n_components=3)
S = ica.fit_transform(X)

plt.figure()
for i in range(S.shape[1]):
    plt.plot(S.T[i])
plt.show()