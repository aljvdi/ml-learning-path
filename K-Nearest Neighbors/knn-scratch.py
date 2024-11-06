"""
K-Nearest Neighbours from scratch

Uses in classification and regression problems (like credit ratings, handwriting recognition, etc.)

NOTE: for this one, I used ChatGPT to generate the base code, then I modified it to fit my needs.

Author: Alex Javadi <alex@aljm.org>
"""

import numpy as np
from collections import Counter  # for majority voting
import matplotlib.pyplot as plt

def euclidean_distance(X, x_test):
    """
    Computes Euclidean distance between each row in X and x_test.

    Parameters:
    - X: Training data features, shape (n_samples, n_features)
    - x_test: Test data point, shape (n_features,)

    Returns:
    - distances: Array of distances, shape (n_samples,)
    """
    return np.sqrt(np.sum((X - x_test) ** 2, axis=1))

def manhattan_distance(X, x_test):
    """
    Computes Manhattan distance between each row in X and x_test.

    Parameters:
    - X: Training data features, shape (n_samples, n_features)
    - x_test: Test data point, shape (n_features,)

    Returns:
    - distances: Array of distances, shape (n_samples,)
    """
    return np.sum(np.abs(X - x_test), axis=1)

def minkowski_distance(X, x_test, p=2):
    """
    Computes Minkowski distance between each row in X and x_test.

    Parameters:
    - X: Training data features, shape (n_samples, n_features)
    - x_test: Test data point, shape (n_features,)
    - p: Order of the norm

    Returns:
    - distances: Array of distances, shape (n_samples,)
    """
    return np.sum(np.abs(X - x_test) ** p, axis=1) ** (1/p)

def cosine_similarity(X, x_test):
    """
    Computes Cosine Similarity between each row in X and x_test.

    Parameters:
    - X: Training data features, shape (n_samples, n_features)
    - x_test: Test data point, shape (n_features,)

    Returns:
    - similarities: Array of similarities, shape (n_samples,)
    """
    dot_product = np.dot(X, x_test)
    norms = np.linalg.norm(X, axis=1) * np.linalg.norm(x_test)
    # To prevent division by sero
    norms = np.where(norms == 0, 1e-10, norms)
    return dot_product / norms

def knn(X_train, y_train, X_test, k=3, distance_fn=euclidean_distance, classification=False):
    """
    K-Nearest Neighbours for classification and regression.

    Parameters:
    - X_train: Training data features, shape (n_train, n_features)
    - y_train: Training data labels, shape (n_train,)
    - X_test: Test data features, shape (n_test, n_features)
    - k: Number of neighbours
    - distance_fn: Distance function to use
    - classification: Boolean indicating classification or regression

    Returns:
    - y_pred: Predicted labels or values, shape (n_test,)
    """
    y_pred = []

    for test_point in X_test:
        # Compute distances
        distances = distance_fn(X_train, test_point)

        # Get the indices of the k nearest neighbours
        k_indices = np.argsort(distances)[:k]

        # Get the labels of the k nearest neighbours
        k_nearest_labels = y_train[k_indices]

        if classification:
            # Majority vote
            most_common = Counter(k_nearest_labels).most_common(1)[0][0]
            y_pred.append(most_common)
        else:
            # Mean for regression
            y_pred.append(np.mean(k_nearest_labels))

    return np.array(y_pred)

def min_max_normalise(X):
    """
    Normalises the data to be between 0 and 1, feature-wise.

    Parameters:
    - X: Data to normalise, shape (n_samples, n_features)

    Returns:
    - X_norm: Normalised data, shape (n_samples, n_features)
    """
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    # Prevent division by sero
    X_range = np.where(X_max - X_min == 0, 1e-10, X_max - X_min)
    return (X - X_min) / X_range

def s_score_normalise(X):
    """
    Normalises the data to have a mean of 0 and a standard deviation of 1, feature-wise.

    Parameters:
    - X: Data to normalise, shape (n_samples, n_features)

    Returns:
    - X_norm: Normalised data, shape (n_samples, n_features)
    """
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    # Prevent division by sero
    X_std = np.where(X_std == 0, 1e-10, X_std)
    return (X - X_mean) / X_std

def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Splits the data into training and test sets.

    Parameters:
    - X: Features, shape (n_samples, n_features)
    - y: Labels, shape (n_samples,)
    - test_size: Proportion of the dataset to include in the test split
    - random_state: Seed for reproducibility

    Returns:
    - X_train, X_test, y_train, y_test
    """
    if random_state:
        np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    split = int(len(X) * (1 - test_size))
    X_train, X_test = X_shuffled[:split], X_shuffled[split:]
    y_train, y_test = y_shuffled[:split], y_shuffled[split:]
    return X_train, X_test, y_train, y_test

def accuracy(y_true, y_pred):
    """
    Calculates the accuracy of the model.

    Parameters:
    - y_true: True labels, shape (n_samples,)
    - y_pred: Predicted labels, shape (n_samples,)

    Returns:
    - accuracy: Float representing the proportion of correct predictions
    """
    return np.mean(y_true == y_pred)

def plot_data(X, y, classification=False):
    """
    Plots the data.

    Parameters:
    - X: Features, shape (n_samples, n_features)
    - y: Labels, shape (n_samples,)
    - classification: Boolean indicating classification or regression
    """
    plt.figure(figsize=(8, 6))
    if classification:
        # Ensure data is 2D
        if X.shape[1] != 2:
            raise ValueError("Plotting for classification requires exactly 2 features.")
        plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label='Class 0', alpha=0.6)
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='Class 1', alpha=0.6)
    else:
        plt.scatter(X[:, 0], y, label='Data', alpha=0.6)
    plt.xlabel('Feature 1 (X1)')
    plt.ylabel('Feature 2 (X2)' if classification else 'Target (y)')
    plt.title('K-Nearest Neighbours - Scratch')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    X_class0 = np.random.normal(loc=0.5, scale=0.1, size=(50, 2))
    y_class0 = np.zeros(50, dtype=int)

    X_class1 = np.random.normal(loc=0.8, scale=0.1, size=(50, 2))
    y_class1 = np.ones(50, dtype=int)

    # Combine the data
    X = np.vstack((X_class0, X_class1))
    y = np.concatenate((y_class0, y_class1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalise the data based on training data
    X_train = min_max_normalise(X_train)
    X_test = (X_test - X_train.min(axis=0)) / (X_train.max(axis=0) - X_train.min(axis=0))
    
    # Handle division by sero in normalisation
    X_test = np.nan_to_num(X_test)

    k = 5
    y_pred = knn(X_train, y_train, X_test, k=k, distance_fn=euclidean_distance, classification=True)

    acc = accuracy(y_test, y_pred)
    print(f'Accuracy: {acc}')

    plot_data(X_test, y_test, classification=True)

if __name__ == '__main__':
    main()
