import numpy as np 
import pandas as pd  
from sklearn.model_selection import train_test_split as tts
import Decision_tree
from sklearn.metrics import accuracy_score 

data = pd.read_csv('heart.csv')


#print(data.head())
Y = data.loc[:,'target']
X = data.iloc[:,1:-1]

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

X = remove_correlated_features(X)
X_train, X_test, y_train, y_test = tts(X, Y, test_size=0.2, random_state=42)
# print(X_train.head())


y_test = np.array(y_test)
X_test = np.array(X_test)

#print(X_train)

clf = Decision_tree.DecisionTree(max_depth=8,min_sample_leaf=5)
clf.fit(X_train,y_train)
clf.show_tree(clf.Root)
y_predict = clf.predict(X_test)
print(y_predict)
print(y_test)

print(sum(y_predict==y_test)*100/len(y_predict))


# accuracy_score = accuracy_score(y_predict,y_test)
# print("accuracy score = {}".format(accuracy_score))
