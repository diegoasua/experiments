import numpy as np
from typing import List, Tuple, Optional

class XGBoostNode:
    def __init__(
        self,
        feature: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional['XGBoostNode'] = None,
        right: Optional['XGBoostNode'] = None,
        gain: float = 0.0,
        value: float = 0.0
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gain = gain
        self.value = value  # Contains the prediction value for leaf nodes

class XGBoostTree:
    def __init__(
        self,
        max_depth: int = 6,
        min_child_weight: float = 1.0,
        gamma: float = 0.0,
        lambda_: float = 1.0,
        epsilon: float = 0.1
    ):
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.gamma = gamma  # Minimum loss reduction for split
        self.lambda_ = lambda_  # L2 regularization
        self.epsilon = epsilon  # Small number to prevent division by zero
        self.root = None

    def _calc_leaf_value(self, grad: np.ndarray, hess: np.ndarray) -> float:
        """Calculate leaf value using the XGBoost formula"""
        return -np.sum(grad) / (np.sum(hess) + self.lambda_ + self.epsilon)

    def _calc_gain(self, grad: np.ndarray, hess: np.ndarray) -> float:
        """Calculate gain for a split using the XGBoost formula"""
        G, H = np.sum(grad), np.sum(hess)
        return (G * G) / (H + self.lambda_ + self.epsilon)

    def _find_best_split(
        self, X: np.ndarray, grad: np.ndarray, hess: np.ndarray
    ) -> Tuple[Optional[int], Optional[float], float]:
        """Find the best split for a node"""
        best_gain = 0.0
        best_feature = None
        best_threshold = None
        n_features = X.shape[1]

        for feature in range(n_features):
            # Sort feature values and get unique thresholds
            thresholds = np.unique(X[:, feature])[:-1]  # Exclude the last value
            
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                # Skip if either child has less than min_child_weight
                if (np.sum(hess[left_mask]) < self.min_child_weight or
                    np.sum(hess[right_mask]) < self.min_child_weight):
                    continue

                gain_left = self._calc_gain(grad[left_mask], hess[left_mask])
                gain_right = self._calc_gain(grad[right_mask], hess[right_mask])
                gain = gain_left + gain_right - self._calc_gain(grad, hess)

                # Apply gamma minimum gain
                if gain < self.gamma:
                    continue

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(
        self, X: np.ndarray, grad: np.ndarray, hess: np.ndarray, depth: int = 0
    ) -> XGBoostNode:
        """Recursively build the XGBoost tree"""
        # Create a leaf node if max depth is reached
        if depth >= self.max_depth:
            return XGBoostNode(value=self._calc_leaf_value(grad, hess))

        # Find the best split
        feature, threshold, gain = self._find_best_split(X, grad, hess)

        # Create a leaf node if no valid split is found
        if feature is None:
            return XGBoostNode(value=self._calc_leaf_value(grad, hess))

        # Split the data
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        # Create child nodes
        left_node = self._build_tree(
            X[left_mask], grad[left_mask], hess[left_mask], depth + 1
        )
        right_node = self._build_tree(
            X[right_mask], grad[right_mask], hess[right_mask], depth + 1
        )

        return XGBoostNode(feature, threshold, left_node, right_node, gain)

    def fit(self, X: np.ndarray, grad: np.ndarray, hess: np.ndarray):
        """Fit the tree to the gradients and hessians"""
        self.root = self._build_tree(X, grad, hess)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions for the input samples"""
        return np.array([self._predict_single(x, self.root) for x in X])

    def _predict_single(self, x: np.ndarray, node: XGBoostNode) -> float:
        """Make prediction for a single sample"""
        if node.feature is None:  # Leaf node
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._predict_single(x, node.left)
        return self._predict_single(x, node.right)

class XGBoostClassifier:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.3,
        min_child_weight: float = 1.0,
        gamma: float = 0.0,
        lambda_: float = 1.0,
        epsilon: float = 1e-7
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.trees: List[XGBoostTree] = []

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))

    def _calculate_gradients(
        self, y: np.ndarray, pred: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate gradients and hessians for binary logistic loss"""
        pred_prob = self._sigmoid(pred)
        grad = pred_prob - y
        hess = pred_prob * (1 - pred_prob)
        return grad, hess

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the XGBoost model"""
        # Initialize predictions with zeros
        pred = np.zeros(len(y))
        
        # Train each tree
        for _ in range(self.n_estimators):
            # Calculate gradients and hessians
            grad, hess = self._calculate_gradients(y, pred)
            
            # Create and train new tree
            tree = XGBoostTree(
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                gamma=self.gamma,
                lambda_=self.lambda_,
                epsilon=self.epsilon
            )
            tree.fit(X, grad, hess)
            
            # Update predictions
            update = tree.predict(X)
            pred += self.learning_rate * update
            
            # Store the tree
            self.trees.append(tree)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        # Get raw predictions
        pred = np.zeros(len(X))
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        
        # Convert to probabilities
        probas = self._sigmoid(pred)
        return np.vstack([1 - probas, probas]).T

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict classes"""
        probas = self.predict_proba(X)
        return (probas[:, 1] >= 0.5).astype(int)