# Machine Learning algorithms from scratch
- # Expectation Maximization Algorithm

![Alt text](./Figure_1.png "Title")
# Decision Trees
```python
import Decision_tree

clf = Decision_tree.DecisionTree(max_depth=8,min_sample_leaf=5)
clf.fit(X_train,y_train)

# This shows the tree that has been built
clf.show_tree(clf.Root)

y_predict = clf.predict(X_test)

```
