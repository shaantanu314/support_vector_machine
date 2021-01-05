from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing

img1 = Image.open('./img1.jpeg')
img2 = Image.open('./img2.jpeg')
x1 = np.asarray(img1)
x1 = x1[:180,:240,:]
x2 = np.asarray(img2)
x2 = x2[:180,:240,:]

x1 = x1.reshape(-1)
x2 = x2.reshape(-1)

# shape is (180,240,3)
# remember for reshaping 


# img = Image.fromarray(x1,'RGB')
# img.show()
# img = Image.fromarray(x2,'RGB')
# img.show()

X = [x1 , x2]
X = np.array(X)

A = [[1,0],[0,1]]
print A
X = np.dot(A,X)
X = np.array(X)

# img = Image.fromarray(X[1].reshape(180,240,3).astype('uint8'),'RGB')
# img.show()

xmix1 = X[0]
xmix2 = X[1]

xmix1 = xmix1 - np.mean(xmix1)
xmix2 = xmix2 - np.mean(xmix2) 
# The first step computes U, which is a rotation matrix by an angle theta, in order to maximize the variance

theta_0 = 0.5*math.atan(-2*(sum(np.multiply(xmix1,xmix2)))/(sum(xmix1**2 - xmix2**2)))
print theta_0*180/(math.pi)
# ~37.59 degree is the angle by which the joint density should be rotated to maximiza variance

U = np.array([[math.cos(theta_0),-math.sin(theta_0)],[math.sin(theta_0),math.cos(theta_0)]])
U = U.T
sigma1 = sum((xmix1*math.cos(theta_0) + xmix2*math.sin(theta_0))**2)
sigma2 = sum((xmix1*math.cos(theta_0-math.pi/2) + xmix2*math.sin(theta_0-math.pi/2))**2)

sig_inv = [[1/math.sqrt(sigma1),0],[0,math.sqrt(sigma2)]]


xmix1 = sig_inv[0][0]*(U[0][0]*xmix1 + U[0][1]*xmix2)
xmix2 = sig_inv[1][1]*(U[1][0]*xmix1 + U[1][1]*xmix2)


phi_0 = 0.25*math.atan(-sum(2*(xmix1**3)*xmix2-2*xmix1*(xmix2**3)) / sum(3*(xmix1**2)*(xmix2**2)-0.5*(xmix1**4)-0.5*(xmix2**4)))
print phi_0*180/math.pi

V = [[math.cos(phi_0),math.sin(phi_0)],[-math.sin(phi_0),math.cos(phi_0)]]
print V

X1 = xmix1
X2 = xmix2

# S1 = preprocessing.normalize((V[0][0]*X1 + V[0][1]*X2).reshape(180,240,3).astype('uint8'))
# S2 = preprocessing.normalize((V[1][0]*X1 + V[1][1]*X2).reshape(180,240,3).astype('uint8'))


S1 = V[0][0]*X1 + V[0][1]*X2
S2 = V[1][0]*X1 + V[1][1]*X2


S1 = S1.reshape(180,240,3)

mx =  np.max(S1,axis=(0,1,2)) 

S1[:,:,0] = S1[:,:,0]/mx*255
S1[:,:,1] = S1[:,:,1]/mx*255
S1[:,:,2] = S1[:,:,2]/mx*255

S2 = S2.reshape(180,240,3)

mx =  np.max(S2,axis=(0,1,2)) 

S2[:,:,0] = S2[:,:,0]/mx*255
S2[:,:,1] = S2[:,:,1]/mx*255
S2[:,:,2] = S1[:,:,2]/mx*255


print S2


img = Image.fromarray((S1.reshape(180,240,3)).astype('uint8'),'RGB')
img.show()
