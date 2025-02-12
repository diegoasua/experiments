import numpy as np
from typing import List, Optional
from collections import Counter

class Node:
    def __init__(
        self,
        feature: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional['Node'] = None,
        right: Optional['Node'] = None,
        value: Optional[float] = None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeRF:
    """Decision Tree modified for Random Forest"""
    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        max_features: Optional[int] = None,
        criterion: str = "gini"
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.root = None

    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Create a bootstrap sample of the dataset"""
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[idxs], y[idxs]

    def _calculate_impurity(self, y: np.ndarray) -> float:
        """Calculate impurity (gini or entropy) of a node"""
        if len(y) == 0:
            return 0
            
        proportions = np.bincount(y) / len(y)
        
        if self.criterion == "gini":
            return 1 - np.sum(proportions ** 2)
        else:  # entropy
            return -np.sum(proportions * np.log2(proportions + 1e-10))

    def _information_gain(self, y: np.ndarray, X_column: np.ndarray, threshold: float) -> float:
        """Calculate information gain for a split"""
        parent_impurity = self._calculate_impurity(y)

        left_mask = X_column <= threshold
        right_mask = ~left_mask

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return 0

        n = len(y)
        n_l, n_r = np.sum(left_mask), np.sum(right_mask)
        
        child_impurity = (n_l / n) * self._calculate_impurity(y[left_mask]) + \
                        (n_r / n) * self._calculate_impurity(y[right_mask])

        return parent_impurity - child_impurity

    def _best_split(self, X: np.ndarray, y: np.ndarray, feature_idxs: np.ndarray) -> tuple:
        """Find the best split considering only a subset of features"""
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature in feature_idxs:
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                gain = self._information_gain(y, X[:, feature], threshold)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """Recursively grow the tree"""
        n_samples, n_features = X.shape

        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           len(np.unique(y)) == 1:
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)

        # Randomly select subset of features
        n_features_split = self.max_features or int(np.sqrt(n_features))
        feature_idxs = np.random.choice(n_features, n_features_split, replace=False)

        # Find best split
        best_feature, best_threshold = self._best_split(X, y, feature_idxs)
        
        if best_feature is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)

        # Create child splits
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(best_feature, best_threshold, left, right)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the decision tree"""
        # Create bootstrap sample
        X_sample, y_sample = self._bootstrap_sample(X, y)
        self.root = self._grow_tree(X_sample, y_sample)
        return self

    def _traverse_tree(self, x: np.ndarray, node: Node) -> int:
        """Traverse the tree to make a prediction"""
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions for samples in X"""
        return np.array([self._traverse_tree(x, self.root) for x in X])

class RandomForest:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        max_features: Optional[int] = None,
        criterion: str = "gini",
        n_jobs: int = 1
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.n_jobs = n_jobs
        self.trees: List[DecisionTreeRF] = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the random forest"""
        self.trees = []
        
        for _ in range(self.n_estimators):
            tree = DecisionTreeRF(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                criterion=self.criterion
            )
            tree.fit(X, y)
            self.trees.append(tree)
            
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the random forest"""
        # Get predictions from all trees
        predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Take majority vote for each sample
        return np.array([
            Counter(predictions[:, i]).most_common(1)[0][0]
            for i in range(X.shape[0])
        ])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        # Get predictions from all trees
        predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Calculate probabilities for each class
        n_samples = X.shape[0]
        n_classes = len(np.unique(predictions))
        probabilities = np.zeros((n_samples, n_classes))
        
        for i in range(n_samples):
            class_counts = Counter(predictions[:, i])
            for class_idx in range(n_classes):
                probabilities[i, class_idx] = class_counts.get(class_idx, 0) / self.n_estimators
                
        return probabilities