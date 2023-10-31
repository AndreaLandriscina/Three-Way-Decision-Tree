# Three-Way-Classifier
A Python library that solve multi-class classification problems with continuous attributes on a dataset. The peculiarity of this library is that this kind of machine-learning model can abstein from making a decision using orthopairs and orthopartitions.
<br />
More info: [Three-Way and Semi-supervised Decision Tree Learning Based on Orthopartitions](https://link.springer.com/chapter/10.1007/978-3-319-91476-3_61) <br>


# How to use
The interface for training and for making the prediction is the same as in the sklearn library. [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#)
```python
from sklearn.datasets import load_breast_cancer as dataset
from sklearn.model_selection import train_test_split
data = dataset()
train_X, X_test, train_y, y_test = train_test_split(data.data, data.target, random_state=1)
clf = ThreeWayClassifier(feature_names=data.feature_names, target_names=data.target_names, max_depth=3, alpha=0.3)
tree = clf.fit(X=train_X, y=train_y)
tree.to_graphviz("cancer.dot", shape='rectangle', sorting=False)
```
![CancerThreeWay](https://github.com/AndreaLandri/Three-Way/assets/70241844/6561240d-27fa-4c92-b6ca-3a759cae1e22)
# Class
```python
class ThreeWayClassifier(epsilon=1, alpha=0.4, max_depth=None, min_samples_split=2, max_leaf_nodes=None, min_mutual_information=-1)
```
- Parameters
  -	epsilon: float, default=1 <br>
  &nbsp;Classification error cost ϵ.
  -	alpha: float, default=0.4 <br>
  &nbsp;Abstention cost α.
  -	max_depth: int, default=None <br>
  &nbsp;Max depth of the tree. 
  -	min_samples_split: int, default=2 <br>
  &nbsp;Minimum number of samples to make a split.
  - max_leaf_nodes: int, default=None <br>
  &nbsp;Max number of leaf nodes. 
  -	min_mutual_information: float, default=-1 <br>
  &nbsp;Minimum value of mutual information to make a split.
- Attributes
  - feature_names: list <br>
  &nbsp;Names of the features.
  - target_names: list <br>
  &nbsp;Names of the targets.
  - tree: Tree (treelib) <br>
  &nbsp;The tree object’s instance of the treelib library.

## Methods 
```python
accuracy(ypred, y_test, weight)
```
- Parameters
  -	ypred: list <br>
  &nbsp;List of expected predictions computed with predict.
  -	y_test: list <br>
  &nbsp;List of class to predict.
  -	weight: float, default = 0.5 <br>
  &nbsp;Value between 0 and 1 determining the degree to which abstention cases are accepted.. 
- Return
  - score: float <br>
  &nbsp;Accuracy percentage.

```python
create_orthopairs(instances, expected_error_cost, majority_class)
```
- Parameters
  -	instances: list <br>
  &nbsp;The set of instance.
  -	expected_error_cost: float <br>
  &nbsp;The expected classification error cost.
  -	majority_class: int <br>
  &nbsp;The majority class of the set of instances. 

```python
create_orthopartition(sub_groups=None, instances=None, instance=None)
```
- Parameters
  -	sub_groups: list, default=None <br>
  &nbsp;The instances divided.
  -	instances: list, default=None <br>
  &nbsp;The set of instances with which to create the orthopartition.
  -	instance: int, default=None <br>
  &nbsp;The instance from which to retrieve the two groups from sub_groups.
- Return
  - Orthopartition <br>
  &nbsp;The created orthopartition.

```python
exec_predict(instance, node=None)
```
- Parameters
  -	instance: list <br>
  &nbsp;The instances divided.
  -	node: Node, default=None <br>
  &nbsp;The current node to be analyzed.
- Return
  - node.data: int <br>
  &nbsp;The predicted class for a certain node.

```python
expected_error_cost(instances)
```
- Parameters
  -	instances: list <br>
  &nbsp;The set of instances.
  -	y: list <br>
  &nbsp;The list of targets.
- Return
  - expected_error_cost: float
  &nbsp;The expected classification error cost.
  - majority_class: int <br>
  &nbsp;The majority class of the set of instances.

```python
fit(X, y, instances=None, parent=-1)
```
- Parameters
  -	X: matrix of shape (|instances|, |attributes|) <br>
  &nbsp;The examples used to train the model. Each example consists of a set of values representing features of the dataset.
  -	y: : array of shape |istanze| <br>
  &nbsp;The target values (class labels) of the instances.
  - instances: list, default=None <br>
  &nbsp;The set of instances of the current node.
  - parent: int, default=-1 <br>
  &nbsp;Index of the node that created the current node. Initially, the node to be analyzed is the root that has no father.
- Return
  - •	self.tree <br>
  &nbsp;The tree created through model training.

```python
get_depth()
```
- Return
  -	self.tree.depth: int <br>
  &nbsp;The tree’s depth.

```python
get_n_leaves()
```
- Return
  -	self.tree.leaves: int <br>
  &nbsp;The number of leaves of the tree.

```python
max_node()
```
- Return
  -	max: int <br>
  &nbsp;Node identifier with the highest index.

```python
orthopartition_GT(instances, y)
```
- Parameters
  -	instances: list <br>
  &nbsp;The set of instances.
  -	y: list <br>
  &nbsp;The set of targets.
- Return
  - Orthopartition <br>
  &nbsp;The Orthopartition relative to ground truth.

```python
predict(instances)
```
- Parameters
  -	instances: list <br>
  &nbsp;List of instances to make the prediction with.
- Return
  - predictions: list <br>
  &nbsp;List with expected class values for each instance.

```python
sub_groups(instances, val_attribute)
```
- Parameters
  -	instances: list <br>
  &nbsp;Set of instances to divide.
  -	val_attribute: array-like <br>
  &nbsp;The set of values of a certain attribute.
- Return
  - sub_groups: list <br>
  &nbsp;List containing two lists for each cell.



