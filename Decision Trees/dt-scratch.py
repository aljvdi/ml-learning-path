"""
Decision Trees from Scratch

Author: Alex Javadi <alex@aljm.org>
"""

import numpy as np
import matplotlib.pyplot as plt

def gini_impurity(y):
    """
    Calculate the Gini impurity of a dataset.
    
    Parameters:
    y (array): Array of target labels.

    Returns:
    float: Gini impurity.
    """
    m = len(y)
    if m == 0:
        return 0
    p = np.bincount(y) / m
    return 1 - np.sum(p ** 2)

def entropy_impurity(y):
    """
    Calculate the entropy of a dataset.
    
    Parameters:
    y (array): Array of target labels.

    Returns:
    float: Entropy.
    """
    m = len(y)
    if m == 0:
        return 0
    p = np.bincount(y) / m
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

def information_gain(left, right, criterion='gini'):
    """
    Calculate the information gain of a split.
    
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
    Build a decision tree classifier.
    
    Parameters:
    X (array): Input features.
    y (array): Target labels.
    max_depth (int): Maximum depth of the tree.
    criterion (str): Impurity criterion ('gini' or 'entropy').

    Returns:
    tuple or int: Tree structure or class label for leaf nodes.
    """
    if max_depth == 0 or len(np.unique(y)) == 1:
        return np.bincount(y).argmax()

    _, n_features = X.shape
    best_gain = 0
    best_feature = None
    best_threshold = None

    for i in range(n_features):
        thresholds = np.unique(X[:, i])
        for t in thresholds:
            left = y[X[:, i] < t]
            right = y[X[:, i] >= t]
            gain = information_gain(left, right, criterion)
            if gain > best_gain:
                best_gain = gain
                best_feature = i
                best_threshold = t

    if best_gain == 0:
        return np.bincount(y).argmax()

    left_idx = X[:, best_feature] < best_threshold
    X_left, y_left = X[left_idx], y[left_idx]
    X_right, y_right = X[~left_idx], y[~left_idx]

    left_branch = decision_tree(X_left, y_left, max_depth - 1, criterion)
    right_branch = decision_tree(X_right, y_right, max_depth - 1, criterion)

    return (best_feature, best_threshold, left_branch, right_branch)

def predict_tree(tree, X):
    """
    Predict target labels for a dataset.
    
    Parameters:
    tree (tuple or int): Decision tree structure.
    X (array): Input features.

    Returns:
    array: Predicted labels.
    """
    if isinstance(tree, (int, np.integer)):
        return np.full(X.shape[0], tree)

    feature, threshold, left_branch, right_branch = tree
    left_idx = X[:, feature] < threshold
    y_pred = np.empty(X.shape[0], dtype=int)
    y_pred[left_idx] = predict_tree(left_branch, X[left_idx])
    y_pred[~left_idx] = predict_tree(right_branch, X[~left_idx])
    return y_pred

def accuracy_score(y_true, y_pred):
    """
    Calculate the accuracy of predictions.
    
    Parameters:
    y_true (array): True labels.
    y_pred (array): Predicted labels.

    Returns:
    float: Accuracy score.
    """
    return np.sum(y_true == y_pred) / len(y_true)

def plot_node(tree, depth=0):
    """
    Recursively plot the nodes of the decision tree.
    
    Parameters:
    tree (tuple or int): Decision tree structure.
    depth (int): Current depth in the tree.
    """
    if isinstance(tree, (int, np.integer)):
        plt.text(0, 0, f'Class {tree}', ha='center', va='center',
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
        return

    feature, threshold, left_branch, right_branch = tree
    plt.text(0, 0, f'Feature {feature}\nThreshold {threshold:.2f}', ha='center', va='center',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    plt.plot([0, 0], [-1, 1], c='k')
    plt.plot([-1, 1], [0, 0], c='k')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plot_node(left_branch, depth + 1)
    plot_node(right_branch, depth + 1)

def plot_tree(tree, X, y):
    """
    Plot the decision tree.
    
    Parameters:
    tree (tuple): Decision tree structure.
    X (array): Input features.
    y (array): Target labels.
    """
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=20)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Tree - Scratch')
    plot_node(tree)
    plt.show()

def main():
    
    # Generate synthetic dataset
    X = np.random.rand(100, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)

    # Train decision tree
    tree = decision_tree(X, y, max_depth=3, criterion='gini')

    # Predict target labels
    y_pred = predict_tree(tree, X)

    # Calculate accuracy
    acc = accuracy_score(y, y_pred)
    print(f'Accuracy: {acc:.2f}')

    # Plot the decision tree
    plot_tree(tree, X, y)

if __name__ == '__main__':
    main()
