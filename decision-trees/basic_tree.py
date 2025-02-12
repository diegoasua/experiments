import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        # Feature index to split on
        self.feature = feature
        # Threshold value for the split
        self.threshold = threshold
        # Left child node
        self.left = left
        # Right child node
        self.right = right
        # Predicted value for leaf nodes
        self.value = value

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, criterion="gini"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None

    def fit(self, X, y):
        self.n_classes = len(np.unique(y)) if self.criterion != "mse" else None
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        
        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           len(np.unique(y)) == 1:
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value)

        # Find the best split
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value)

        # Create child splits
        left_idxs = X[:, best_feature] < best_threshold
        right_idxs = ~left_idxs
        
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)

        return Node(best_feature, best_threshold, left, right)

    def _best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                gain = self._information_gain(y, X[:, feature], threshold)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, y, X_column, threshold):
        parent_impurity = self._calculate_impurity(y)

        left_idxs = X_column < threshold
        right_idxs = ~left_idxs
        
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(y[left_idxs]), len(y[right_idxs])
        
        if n_l == 0 or n_r == 0:
            return 0

        child_impurity = (n_l / n) * self._calculate_impurity(y[left_idxs]) + \
                        (n_r / n) * self._calculate_impurity(y[right_idxs])

        return parent_impurity - child_impurity

    def _calculate_impurity(self, y):
        if self.criterion == "gini":
            return self._gini(y)
        elif self.criterion == "entropy":
            return self._entropy(y)
        else:  # MSE for regression
            return np.var(y)

    def _gini(self, y):
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum(proportions ** 2)

    def _entropy(self, y):
        proportions = np.bincount(y) / len(y)
        return -np.sum(proportions * np.log2(proportions + 1e-10))

    def _calculate_leaf_value(self, y):
        if self.criterion == "mse":
            return np.mean(y)
        else:  # Classification
            return Counter(y).most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature] < node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)