# Machine Learning algorithms from scratch
- # K-Nearest Neighbour
K-Nearest Neighbour is one of the simplest Machine Learning algorithms based on Supervised Learning technique. K-NN algorithm assumes the similarity between the new case/data and available cases and put the new case into the category that is most similar to the available categories.
## Usage
# K nearest neighbours
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