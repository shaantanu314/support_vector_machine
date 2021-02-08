import numpy as np
import Expectation_maximization as EM
import matplotlib.pyplot as plt

X = [ [np.random.normal()*20+10,np.random.normal()*20+30] for i in range(100)]
Y = [[np.random.normal()*10+70,np.random.normal()*10+90] for i in range(100)] 
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
