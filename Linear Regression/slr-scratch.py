"""
Simple Linear Regression from scratch

Author: Alex Javadi <alex@aljm.org>
"""

import numpy as np
import matplotlib.pyplot as plt

def mse(y_true, y_pred):
    """
    Mean Squared Error (MSE) is a measure of the average of the squares of the errors or deviations.
    """
    return np.mean((y_true - y_pred)**2)

def slr(x, y):
    """    
    Simple Linear Regression (SLR) is a fundamental statistical method used to model the linear relationship between a dependent variable y and a single independent variable x.
    The primary goal is to find the best-fitting straight line through the data points that minimizes the discrepancy between the observed values and the values predicted by the line.

    The equation of a straight line is given by:
    y = mx + c
    where:
        - y is the dependent variable
        - x is the independent variable
        - m is the slope of the line

    """

    # Calculate the mean of x and y
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Calculate the slope (m) of the line
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean)**2)
    m = numerator / denominator

    # Calculate the y-intercept (c) of the line
    c = y_mean - m * x_mean

    return m, c

def slr_fit(x, y):
    """
    Fit the data to the SLR model
    """
    m, c = slr(x, y)
    y_pred = m * x + c
    return y_pred

def predict(x, m, c):
    """
    Predict the dependent variable using the SLR model
    """
    return m * x + c

def plot(x, y, y_pred):
    """
    Plot the data and the SLR model
    """
    plt.scatter(x, y, color='blue', label='Data')
    plt.plot(x, y_pred, color='red', label='SLR Model')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Simple Linear Regression')
    plt.legend()
    plt.show()

def main():
    # Set a random seed for reproducibility
    np.random.seed(1)

    # Generate random data
    x = np.array([np.random.randint(0, 100) for i in range(100)])
    y = np.array([2 * i + 1 + np.random.randint(-10, 10) for i in x])

    # Fit the data to the SLR model
    y_pred = slr_fit(x, y)

    # Calculate the Mean Squared Error (MSE)
    mse_value = mse(y, y_pred)
    print(f'Mean Squared Error (MSE): {mse_value}')

    # Plot the data and the SLR model
    plot(x, y, y_pred)

def test_cases():
    assert mse(np.array([1, 2, 3]), np.array([1, 2, 3])) == 0
    assert mse(np.array([1, 2, 3]), np.array([2, 3, 4])) == 1

if __name__ == '__main__':
    test_cases()
    main()