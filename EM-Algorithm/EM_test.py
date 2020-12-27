import numpy as np
import Expectation_maximization as EM
import matplotlib.pyplot as plt

X = [ [np.random.rand()*10+10,np.random.rand()*50+10] for i in range(500)]
Y = [[np.random.rand()*50+10,np.random.rand()*10+10] for i in range(150)] 
# Z = [[np.random.rand()*10+30,np.random.rand()*10*10] for i in range(20)] 
X = np.array(X)
Y = np.array(Y)
# Z = np.array(Z)
X = np.append(X,Y,axis=0)
# X = np.append(X,Z,axis=0)

plt.figure()
plt.scatter(X[:,0],X[:,1])
plt.show()
clf = EM.EM_Model(X,2,iter=10)
clf.fit()
clf.plot_contour()
