import Orthopartition
import numpy as np
from treelib import Tree

class ThreeWayClassifier:
    def __init__(self, feature_names, target_names, epsilon=1, alpha=0.4, max_depth=None, min_samples_split=2, min_mutual_information=0.0, max_leaf_nodes=None):
        # error cost
        if epsilon <= alpha:
            raise Exception("epsilon can not be lower than alpha")
        if epsilon <= 0 or alpha <= 0:
            raise Exception(
                "epsilon and/or alpha can not lower or equal than 0")
        self.epsilon = epsilon
        # abstention cost
        self.alpha = alpha
        if max_depth is not None and max_depth <= 0:
            raise Exception("max_depth can not be lower than 1")
        self.max_depth = max_depth
        self.feature_names = feature_names
        self.target_names = target_names
        if min_samples_split < 0:
            raise Exception("min_samples_split can not be lower than 0")
        self.min_samples_split = int(min_samples_split)
        self.min_mutual_information = min_mutual_information
        if max_leaf_nodes is not None and max_leaf_nodes <= 0:
            raise Exception("max_depth can not be lower than 1")
        self.max_leaf_nodes = max_leaf_nodes
        
        self.targets = list()
        for x, value in enumerate(target_names):
            self.targets.append(x)
        self.tree = Tree()

    def orthopartition_GT(self, instances, y):
        # orthopartition ground truth
        orthopairs_gt = list()

        for j in range(len(self.targets)):
            temp = list()
            temp.append(set())  # P
            temp.append(set())  # N
            temp.append(set())  # Bnd
            orthopairs_gt.append(temp)
        for target in range(len(self.targets)):
            for i in range(len(instances)):
                if y[i] == target:
                    orthopairs_gt[target][0].add(instances[i])
                else:
                    orthopairs_gt[target][1].add(instances[i])
        lst = list()
        for orthopair in orthopairs_gt:
            lst.append(Orthopartition.Orthopair.Orthopair(
                p=orthopair[0], n=orthopair[1], bnd=orthopair[2]))
        return Orthopartition.Orthopartition(family=lst)

    def sub_groups(self, instances, attribute):
        # for each element x create two sub-groups (<= x and > x)
        sub_groups = list()
        for instance in instances:
            tmpList = list()
            lower = list()
            higher = list()
            for instance1 in instances:
                if attribute[instance1] <= attribute[instance]:
                    lower.append(instance1)
                else:
                    higher.append(instance1)
            tmpList.append(lower)
            tmpList.append(higher)
            sub_groups.append(tmpList)
        return sub_groups

    def create_orthopairs(self, expected_error_cost, group, class_magg):
        # create list of orthopair
        if expected_error_cost < len(group)*self.alpha:
            for x in group:
                self.orthopairs[class_magg][0].add(x)
                for el in range(len(self.target_names)):
                    if el != class_magg:
                        self.orthopairs[el][1].add(x)
        else:
            for x in group:
                for el in self.targets:
                    self.orthopairs[el][2].add(x)

    def expected_error_cost(self, instances, y):
        lst_cont_inst = list()
        for el in range(len(self.target_names)):
            lst_cont_inst.append(0)
        for x in instances:
            lst_cont_inst[y[x]] += 1

        value_class_magg = np.max(lst_cont_inst)
        class_magg = lst_cont_inst.index(value_class_magg)

        num_ist_min = np.sum(lst_cont_inst) - value_class_magg
        expected_error_cost = self.epsilon * num_ist_min
        return expected_error_cost, class_magg

    def create_orthopartition(self, y, sub_groups=None, instances=None, instance=None):
        self.orthopairs = list()
        for j in range(len(self.targets)):
            temp = list()
            temp.append(set())  # P
            temp.append(set())  # N
            temp.append(set())  # Bnd
            self.orthopairs.append(temp)
        # orthopairs for the computation of the mutual information
        if instance is not None:
            lower = sub_groups[instance][0]
            higher = sub_groups[instance][1]
            expected_error_cost, class_magg = self.expected_error_cost(lower, y)
            self.create_orthopairs(expected_error_cost, lower, class_magg)
            expected_error_cost, class_magg = self.expected_error_cost(higher, y)
            self.create_orthopairs(expected_error_cost, higher, class_magg)
        # orthopairs for the computation of the orthopartition of a leaf
        elif instances is not None:
            expected_error_cost, class_magg = self.expected_error_cost(
                instances, y)
            self.create_orthopairs(expected_error_cost, instances, class_magg)
        lst = list()
        for orthopair in self.orthopairs:
            lst.append(Orthopartition.Orthopair.Orthopair(
                p=orthopair[0], n=orthopair[1], bnd=orthopair[2]))
        return Orthopartition.Orthopartition(family=lst)

    def max_node(self):
        max = 0
        nodes = self.tree.all_nodes()
        for node in nodes:
            if node.identifier > max:
                max = node.identifier
        return max

    def create_leaf(self, y, instances, parent, mutual_info = None):
        o = self.create_orthopartition(y, instances=instances)
        identifier = self.max_node() + 1
        if len(o.family[0].bnd) == 0:
            for ind in range(len(o.family)):
                if len(o.family[ind].p) != 0:
                    break
            lst_inst = list()
            for i in range(len(set(y))):
                lst_inst.append(0)
            for x in instances:
                lst_inst[y[x]] += 1
                # leaf node
            if mutual_info is None:
                mutual_info = "None"
            self.tree.create_node("node " + "#" + str(identifier) + "\nmutual information = " + str(mutual_info) + "\nsamples = " + str(len(instances)) +
                                  "\nvalue = " + str(lst_inst) + "\n class = " + str(self.target_names[ind]), identifier, parent, ind)
        else:
            # leaf node
            self.tree.create_node(
                "node #" + str(identifier) + " abstention", identifier, parent, -1)
        print("node=" + str(identifier) + " identifier=",
              str(identifier) + " parent=" + str(parent))
        print("----------------")

    def fit(self, X, y, instances=None, parent=-1):
        if instances is None:
            instances = [i for i in range(0, len(X))]

        if len(instances) == 0:
            print("all data considered")
            return

        targets = set()
        for el in instances:
            targets.add(y[el])
        # all the instances have the same target
        if len(targets) == 1:
            # leaf node
            identifier = self.max_node() + 1
            target = targets.pop()
            lst_inst = list()
            for i in range(len(set(y))):
                lst_inst.append(0)
            for x in instances:
                lst_inst[y[x]] += 1
            print("leaf")
            print("identifier:" + str(identifier) + " parent:" + str(parent))
            self.tree.create_node("node" + " #"+str(identifier) + "\nmutual information = 0.0\nsamples = " + str(
                len(instances)) + "\nvalue = " + str(lst_inst) + "\nclass = " + self.target_names[target], identifier, parent, target)
            print("----------------")
            return

        if self.tree.depth(node=self.tree.get_node(parent)) + 1 == self.max_depth or len(instances) < self.min_samples_split or len(self.tree.leaves()) == self.max_leaf_nodes:
            self.create_leaf(y, instances, parent)
            return

        ig_attr_value = list()
        # for each feature compute the corresponding mutual information
        orthopartition_gt = self.orthopartition_GT(instances, y)
        for i in range(X.shape[1]):
            ig_attr_value.append(list())
            sub_groups = self.sub_groups(instances, X[:, i])
            # for each instance i1 in the feature i compute the corrisponding orthopartition
            for i1 in range(len(instances)):
                o = self.create_orthopartition(y, sub_groups, instance=i1)
                value = o.mutual_information(orthopartition_gt)
                ig_attr_value[i].append(value)

        lst_max_ind = list()
        lst_max_ig = list()
        for i in range(len(ig_attr_value)):
            lst_max_ind.append(ig_attr_value[i].index(
                np.max(ig_attr_value[i])))
            lst_max_ig.append(np.max(ig_attr_value[i]))

        print("mutual information per feature: ", lst_max_ig)
        stop = False
        if np.max(lst_max_ig) < self.min_mutual_information:
            stop = True
        # print("istances: ", lst_max_ind)
        ig_not_null = False
        for val in lst_max_ig:
            if val != 0:
                ig_not_null = True
                break
        if ig_not_null is False or stop is True:
            print("no information gain or information gain is too low")
            self.create_leaf(y, instances, parent, np.around(max(lst_max_ig), decimals=3))
            return
        ind_feature = lst_max_ig.index(np.max(lst_max_ig))
        ind_ist = lst_max_ind[ind_feature]
        print("split instance: ", ind_ist)
        print("split feature: ", ind_feature)
        val_ist = X[instances[ind_ist], ind_feature]
        print("split value: ", val_ist)

        lst_inst = list()
        for i in range(len(set(y))):
            lst_inst.append(0)
        for x in instances:
            lst_inst[y[x]] += 1

        # root
        if parent == -1:
            identifier = 0
            tag = "node " + "#"+str(identifier) + "\n" + str(self.feature_names[ind_feature]) + " <= " + str(
                val_ist) + "\nmutual information = " + str(np.around(max(lst_max_ig), decimals=3)) + "\nsamples = " + str(len(instances)) + "\nvalue = " + str(lst_inst)
            self.tree.create_node(tag, identifier, None,
                                  (ind_feature, val_ist))
        # branch node
        else:
            identifier = self.max_node() + 1
            tag = "node " + "#"+str(identifier) + "\n" + str(self.feature_names[ind_feature]) + " <= " + str(
                val_ist) + "\nmutual information = " + str(np.around(max(lst_max_ig), decimals=3)) + "\nsamples = " + str(len(instances)) + "\nvalue = " + str(lst_inst)
            self.tree.create_node(tag, identifier, parent,
                                  (ind_feature, val_ist))
        print("node=" + str(identifier) + " identifier=" +
              str(identifier) + " parent=" + str(parent))
        print("----------------")

        lower = list()
        higher = list()
        for instance in instances:
            if X[instance, ind_feature] <= val_ist:
                lower.append(instance)
            else:
                higher.append(instance)
        self.fit(X, y, instances=lower, parent=identifier)
        self.fit(X, y, instances=higher, parent=identifier)
        return self.tree

    def _exec_predict(self, instance, node=None):
        if self.tree is None:
            raise Exception("the model must be trained first")
        if len(instance) != len(self.feature_names):
            raise Exception(
                "the number of values has to be equal the number of feature of the dataset")
        if node is None:
            node = self.tree.get_node(self.tree.root)
        if node.is_leaf():
            return node.data
        ind_feature = node.data[0]
        split_value = node.data[1]
        successors = node.successors(self.tree.identifier)
        if instance[ind_feature] <= split_value:
            node = self.tree.get_node(successors[0])
            return self.exec_predict(instance, node)
        else:
            node = self.tree.get_node(successors[1])
            return self.exec_predict(instance, node)

    def predict(self, instances):
        predictions = np.zeros(len(instances), dtype=int)
        for i, instance in enumerate(instances, 0):
            predictions[i] = self.exec_predict(instance)
        return predictions

    def get_n_leaves(self):
        return len(self.tree.leaves())

    def get_depth(self):
        return self.tree.depth()

    def accuracy(self, ypred, y_test, weight=0.5):
        if weight < 0 or weight > 1:
            raise Exception("weight must be between 0 and 1")
        count_correct = 0
        count_abstention = 0
        for count, pred in enumerate(ypred, start=0):
            if pred == y_test[count]:
                count_correct += 1
            elif pred == -1:
                count_abstention += 1
        print(count_abstention)
        score = (count_correct + weight*count_abstention)/len(ypred)
        return score
