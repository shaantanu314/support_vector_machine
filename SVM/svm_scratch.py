import numpy as np 
import pandas as pd  
 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score 
from sklearn.utils import shuffle
import svm_sk

data = pd.read_csv('./data.csv')
# print(data.head())
diagnosis_map = {'M':1, 'B':-1}
data['diagnosis'] = data['diagnosis'].map(diagnosis_map)
data.drop(data.columns[[-1, 0]], axis=1, inplace=True)




def remove_correlated_features(X):
    corr_threshold = 0.9
    corr = X.corr()
    drop_columns = np.full(corr.shape[0], False, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if corr.iloc[i, j] >= corr_threshold:
                drop_columns[j] = True
    columns_dropped = X.columns[drop_columns]
    X.drop(columns_dropped, axis=1, inplace=True)
    return X



Y = data.loc[:,'diagnosis']
X = data.iloc[:,1:]

# normalization is necessary to bring the data points in 
# a standard range of [0,1] or  say [-1,1] so that the 
# learning also speeds up
X_normalized = MinMaxScaler().fit_transform(X.values)
X = pd.DataFrame(X_normalized)
X.insert(loc=len(X.columns), column='intercept', value=1)
remove_correlated_features(X)

X_train, X_test, y_train, y_test = tts(X, Y, test_size=0.2, random_state=42)

reg_strength = 10000 # regularization strength
learning_rate = 0.000001


cf = svm_sk.Support_Vector_Machine()
W = cf.fit(X_train.to_numpy(),y_train.to_numpy())

y_predict = cf.predict(X_test)
accuracy_score = accuracy_score(y_predict,y_test)
print("accuracy score = {}".format(accuracy_score))