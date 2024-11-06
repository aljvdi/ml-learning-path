"""
Binary Logistic Regression using sklearn

Author: Alex Javadi <alex@aljm.org>
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    # Generate some data
    X = np.random.randn(100, 2).reshape(-1, 2)
    y = np.random.randint(0, 2, 100).reshape(-1, 1)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Fit the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

    # Plot the decision boundary
    x1 = np.linspace(-3, 3, 100)
    x2 = np.linspace(-3, 3, 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = model.predict(np.c_[X1.ravel(), X2.ravel()])
    Z = Z.reshape(X1.shape)

    plt.contourf(X1, X2, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Binary Logistic Regression - sklearn')
    plt.show()