"""
Multiple Linear Regression using scikit-learn

Author: Alex Javadi <alex@aljm.org>
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error # MSE is deprecated
import matplotlib.pyplot as plt

# Generate some random data
X = np.random.rand(100, 5)
y = np.dot(X, np.array([1, 2, 3, 4, 5])) + np.random.rand(100)

model = LinearRegression()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

def plot():
    # Plot the data
    plt.scatter(X[:, 0], y, color='black', label='blue')
    plt.scatter(X[:, 0], model.predict(X), color='red', label='Predictions')
    plt.title('Multiple Linear Regression using scikit-learn')
    plt.show()


def test_cases():
    # Test the model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(root_mean_squared_error(y_test, y_pred))
    
    assert rmse > 0, 'Root Mean Squared Error should be greater than 0'

def main():
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)
    rmse = np.sqrt(root_mean_squared_error(y_test, y_pred))
    print(f'Root Mean Squared Error: {rmse}')

    plot()

if __name__ == '__main__':

    test_cases()
    main()
