# Machine Learning algorithms from scratch
- # Decision Trees

This Decision tree classifier has been implemented for binary classification of data.
# General coment on decision trees
# Disadvantage
- overfitting
# Advantages
-A significant advantage of a decision tree is that it forces the consideration of all possible outcomes of a decision and traces each path to a conclusion. It creates a comprehensive analysis of the consequences along each branch and identifies decision nodes that need further analysis.
(The data set which is used here is for heart disease prediction and even after quite a lot of tweaking it gives accuracy upto 50-52% only )
## Usage
# Decision Trees
```python
import Decision_tree

clf = Decision_tree.DecisionTree(max_depth=8,min_sample_leaf=5)
clf.fit(X_train,y_train)

# This shows the tree that has been built
clf.show_tree(clf.Root)

y_predict = clf.predict(X_test)

```
