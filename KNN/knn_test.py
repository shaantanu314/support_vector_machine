import numpy as np 
import pandas as pd  
from sklearn.model_selection import train_test_split as tts
import knn_sk
from sklearn.metrics import accuracy_score 

data = pd.read_csv('./Iris.csv')

diagnosis_map = {'Iris-setosa':1, 'Iris-versicolor':2 , 'Iris-virginica':3}
data['Species'] = data['Species'].map(diagnosis_map)
# print(data.head())

Y = data.loc[:,'Species']
X = data.iloc[:,1:-1]
# print(X.head())

X_train, X_test, y_train, y_test = tts(X, Y, test_size=0.2, random_state=42)
y_test = np.array(y_test)

clf = knn_sk.KNN(X_train,y_train,7,classification=True)
y_predict = clf.knn(X_test)


accuracy_score = accuracy_score(y_predict,y_test)
print("accuracy score = {}".format(accuracy_score))
