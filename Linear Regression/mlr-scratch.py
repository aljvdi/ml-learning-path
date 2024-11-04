"""
Multiple Linear Regression from scratch

Author: Alex Javadi <alex@aljm.org>
"""

import numpy as np
import matplotlib.pyplot as plt

def mlr(X, y, epochs=1000, lr=0.01):
    """
    Multiple Linear Regression is a statistical technique that models the relationship between two or more independent (predictor) variables and a single dependent (response) variable by fitting a linear equation to observed data. While simple linear regression deals with one predictor, MLR extends this to multiple predictors, allowing for more complex and realistic modeling of real-world phenomena.

    Formula:
        y = b0 + b1*x1 + b2*x2 + ... + bn*xn + e

    Parameters:
    X: np.array
        The independent variables.
    y: np.array
        The dependent variable.
    epochs: int
        The number of iterations.
    lr: float
        The learning rate.

    """
    # Add a column of ones to X
    ones = np.ones((X.shape[0], 1))
    X = np.hstack((ones, X))

    # Initialise the weights
    weights = np.random.rand(X.shape[1])

    # Perform gradient descent
    for _ in range(epochs):
        y_pred = np.dot(X, weights)
        error = y_pred - y
        weights -= lr * (2/X.shape[0]) * np.dot(X.T, error).flatten()

    return weights

def mlr_fit(X, y, epochs=1000, lr=0.01):
    """
    Fit the model and return the predictions.
    """
    weights = mlr(X, y, epochs, lr)

    ones = np.ones((X.shape[0], 1))
    X = np.hstack((ones, X))

    return np.dot(X, weights)

def predict(X, weights):
    """
    Predict the output.
    """
    ones = np.ones((X.shape[0], 1))
    X = np.hstack((ones, X))

    return np.dot(X, weights)

def plot(X, y, y_pred):
    """
    Plot the data.
    """
    plt.scatter(range(len(y)), y, color='blue', label='Actual')
    plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted')
    plt.xlabel('Index')
    plt.ylabel('y')
    plt.title('Multiple Linear Regression')
    plt.legend()
    plt.show()

def plot_residuals(y, y_pred):
    """
    Plot the residuals.
    """
    residuals = y - y_pred
    plt.scatter(y_pred, residuals, color='blue')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('y_pred')
    plt.ylabel('Residuals')
    plt.title('Residuals')
    plt.show()

def mse(y, y_pred):
    """
    Calculate the mean squared error.
    """
    return np.mean((y - y_pred)**2)

def main(X, y):
    y_pred = mlr_fit(X, y)
    plot(X, y, y_pred)
    print('Mean Squared Error:', mse(y, y_pred))
    plot_residuals(y, y_pred)

def test_cases(X, y):

    # Fit the model
    y_pred = mlr_fit(X, y)

    assert mse(y, y_pred) > 0

    weights = mlr(X, y)
    y_pred = predict(X, weights)

    assert mse(y, y_pred) > 0

if __name__ == '__main__':
    # Create a simple dataset
    X = np.random.rand(100, 2)
    y = 2*X[:,0] + 3*X[:,1] + 4 + np.random.randn(100)

    test_cases(X, y)
    main(X,y)
    

