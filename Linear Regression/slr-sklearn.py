"""
Simple Linear Regression using scikit-learn

Author: Alex Javadi <alex@aljm.org>
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error # MSE is deprecated
import matplotlib.pyplot as plt

# Generate some random data
np.random.seed(0)
x = 2 * np.random.rand(100, 1)
y = 5 + 3 * x + np.random.randn(100, 1)

# Scikit-learn Linear Regression model
model = LinearRegression()

def test_cases():
    model.fit(x, y)
    y_pred = model.predict(x)
    rmse = root_mean_squared_error(y, y_pred)

    print(rmse)

    assert np.allclose(y_pred, model.predict(x))
    assert rmse >= 0, 'Root Mean Squared Error should be positive'
    # I am not sure how to test the RMSE
    


def main():
    model.fit(x, y)
    y_pred = model.predict(x)

    # Calculate the Root Mean Squared Error (RMSE)
    rmse = np.sqrt(root_mean_squared_error(y, y_pred))
    print(f'Root Mean Squared Error: {rmse}')

    # Plot the data and the Linear Regression model
    plt.scatter(x, y, color='blue')
    plt.plot(x, y_pred, color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Simple Linear Regression (scikit-learn)')
    plt.show()

if __name__ == '__main__':
    test_cases()
    main()