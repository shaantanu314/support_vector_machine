# Machine Learning algorithms from scrath
- # SVM using stochastic gradient descent
- # K- nearest neighbours


## Usage
# SVM
```python
import svm_sk
clf = svm_sk.Support_Vector_Machine()
clf.fit(X_train,Y_train) 
clf.predict(X_test) 

```
# KNN
```python
import knn_sk


# syntax
# clf = knn_sk.KNN(X_train,Y_train,k,regression=False,classification=False):

# If classification model
clf = knn_sk.KNN(X_train,y_train,7,classification=True)
y_predict = clf.knn(X_test)

# If regression model
clf = knn_sk.KNN(X_train,y_train,7,regression=True)
y_predict = clf.knn(X_test)


```

## Reference
- [SVM from scratch](https://towardsdatascience.com/svm-implementation-from-scratch-python-2db2fc52e5c2#72a3
)

- [KNN ](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761))
    -- [choosing K value for KNN](https://discuss.analyticsvidhya.com/t/how-to-choose-the-value-of-k-in-knn-algorithm/2606/5)