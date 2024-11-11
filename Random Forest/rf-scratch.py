"""
Random Forest Implementation from Scratch

Author: Alex Javadi <alex@aljm.org>
"""

import numpy as np
import matplotlib.pyplot as plt

def gini_impurity(y):
    """
    Calculate the Gini impurity for a set of labels.
    """
    n = len(y)
    if n == 0:
        return 0
    p = np.bincount(y) / n
    return 1 - np.sum(p ** 2)

def entropy_impurity(y):
    """
    Calculate the entropy for a set of labels.
    """
    n = len(y)
    if n == 0:
        return 0
    p = np.bincount(y) / n
    return -np.sum(p * np.log2(p[p > 0]))

def mse(y):
    """
    Calculate the mean squared error for a set of labels (used in regression).
    """
    if len(y) == 0:
        return 0
    return np.var(y)

def information_gain(left, right, criterion='gini'):
    """
    Calculate information gain for a split.
    
    Parameters:
        left (array): Left split of the target labels.
        right (array): Right split of the target labels.
        criterion (str): Criterion for impurity ('gini' or 'entropy').
    
    Returns:
        float: Information gain of the split.
    """
    m = len(left) + len(right)
    parent = np.concatenate([left, right])

    if criterion == 'gini':
        gain = gini_impurity(parent) - (
            (len(left) * gini_impurity(left) + len(right) * gini_impurity(right)) / m
        )
    else:
        gain = entropy_impurity(parent) - (
            (len(left) * entropy_impurity(left) + len(right) * entropy_impurity(right)) / m
        )
    return gain

def decision_tree(X, y, max_depth=3, criterion='gini'):
    """
    Build a decision tree classifier recursively.
    
    Parameters:
        X (array): Input features.
        y (array): Target labels.
        max_depth (int): Maximum depth of the tree.
        criterion (str): Impurity criterion ('gini' or 'entropy').
    
    Returns:
        tuple or int: Tree structure or class label for leaf nodes.
    """
    # Base cases for recursion
    if max_depth == 0 or len(np.unique(y)) == 1:
        return np.bincount(y).argmax()

    best_gain = 0
    best_feature = None
    best_threshold = None

    # Loop through each feature to find the best split
    for i in range(X.shape[1]):
        thresholds = np.unique(X[:, i])
        for t in thresholds:
            left, right = y[X[:, i] < t], y[X[:, i] >= t]
            gain = information_gain(left, right, criterion)
            if gain > best_gain:
                best_gain, best_feature, best_threshold = gain, i, t

    # If no gain, return majority class
    if best_gain == 0:
        return np.bincount(y).argmax()

    left_idx = X[:, best_feature] < best_threshold
    X_left, y_left = X[left_idx], y[left_idx]
    X_right, y_right = X[~left_idx], y[~left_idx]

    left_branch = decision_tree(X_left, y_left, max_depth - 1, criterion)
    right_branch = decision_tree(X_right, y_right, max_depth - 1, criterion)

    return (best_feature, best_threshold, left_branch, right_branch)

def random_forest(X, y, n_trees=10, n_samples=None, n_features=None, max_depth=None):
    """
    Build a random forest classifier.
    
    Parameters:
        X (array): Input features.
        y (array): Target labels.
        n_trees (int): Number of trees in the forest.
        n_samples (int): Number of samples per tree.
        n_features (int): Number of features per split.
        max_depth (int): Maximum depth for each tree.
    
    Returns:
        list: List of decision trees.
    """
    trees = []
    n_samples = n_samples or len(X)
    n_features = n_features or int(np.sqrt(X.shape[1]))

    for _ in range(n_trees):
        idx = np.random.choice(len(X), size=n_samples, replace=True)
        X_sample, y_sample = X[idx], y[idx]
        feature_idx = np.random.choice(X.shape[1], size=n_features, replace=False)
        X_sample = X_sample[:, feature_idx]

        tree = decision_tree(X_sample, y_sample, max_depth, criterion='gini')
        trees.append((tree, feature_idx))
    return trees

def predict_tree(tree, x):
    """
    Predict class label for a single sample based on a single decision tree.
    """
    if not isinstance(tree, tuple):
        return tree
    feature, threshold, left, right = tree
    return predict_tree(left if x[feature] < threshold else right, x)

def predict_forest(trees, X):
    """
    Predict class labels for multiple samples using the forest (ensemble of trees).
    
    Parameters:
        trees (list): List of decision trees.
        X (array): Input features.
    
    Returns:
        array: Predicted class labels.
    """
    predictions = []
    for x in X:
        tree_votes = [predict_tree(tree, x[features]) for tree, features in trees]
        predictions.append(np.bincount(tree_votes).argmax())
    return np.array(predictions)

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def main():
    # Generate random data
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)

    # Split into train and test sets
    train_idx = np.random.rand(len(X)) < 0.8
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[~train_idx], y[~train_idx]

    # Train the random forest
    trees = random_forest(X_train, y_train, n_trees=100, n_samples=50, n_features=3, max_depth=3)

    # Make predictions
    y_pred = predict_forest(trees, X_test)

    # Calculate accuracy
    print(f'Accuracy: {accuracy(y_test, y_pred)}')

    # Plot the decision boundary
    plt.figure(figsize=(8, 6))
    xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    X_plot = np.c_[xx.ravel(), yy.ravel(), np.zeros((xx.size, X_train.shape[1] - 2))]
    Z = predict_forest(trees, X_plot).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', edgecolor='k', s=20)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Random Forest - Scratch')
    plt.show()

if __name__ == '__main__':
    main()        
