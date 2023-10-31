# Three-Way-Classifier
A Python library to solve multi-class classification problems with continuous attributes on a dataset. The peculiarity of the project is that this kind of machine-learning model can abstein from making a decision.
<br />
More info: [Three-Way and Semi-supervised Decision Tree Learning Based on Orthopartitions](https://link.springer.com/chapter/10.1007/978-3-319-91476-3_61)

```python
class ThreeWayClassifier(epsilon=1, alpha=0.4, max_depth=None, min_samples_split=2, max_leaf_nodes=None, min_mutual_information=-1)
```

The interface for training and for making the prediction is the same as in the sklearn library. [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#)

# How to use
```python
from sklearn.datasets import load_breast_cancer as dataset
from sklearn.model_selection import train_test_split
data = dataset()
train_X, X_test, train_y, y_test = train_test_split(data.data, data.target, random_state=1)
clf = ThreeWayClassifier()
tree = clf.fit(X=train_X, y=train_y)
tree.to_graphviz("cancer.dot", shape='rectangle', sorting=False)
```
![CancerThreeWay](https://github.com/AndreaLandri/Three-Way/assets/70241844/6561240d-27fa-4c92-b6ca-3a759cae1e22)

