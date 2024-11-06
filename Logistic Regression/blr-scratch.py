"""
Binary Logistic Regression from scratch

Uses for predicts the probability (spam vs non-spam)

Author: Alex Javadi <alex@aljm.org>
"""

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def log_loss(y, y_hat):
    """
    y: true labels
    y_hat: predicted labels
    """
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

def gradient_descent(X, y, w, b, lr, epochs):
    """
    X: input data
    y: target labels
    w: weights
    b: bias
    lr: learning rate
    epochs: number of iterations
    """
    m = X.shape[0]
    for epoch in range(epochs):
        z = np.dot(X, w) + b
        y_hat = sigmoid(z)
        loss = log_loss(y, y_hat)
        dw = (1 / m) * np.dot(X.T, (y_hat - y))
        db = (1 / m) * np.sum(y_hat - y)
        w -= lr * dw
        b -= lr * db
        if epoch % 100 == 0:
            print(f'Epoch {epoch}: loss={loss}')
    return w, b

def fit(X, y, lr=0.01, epochs=1000):
    """
    X: input data
    y: target labels
    lr: learning rate
    epochs: number of iterations
    """
    n = X.shape[1]
    w = np.zeros(n)
    b = 0
    w, b = gradient_descent(X, y, w, b, lr, epochs)
    return w, b

def predict(X, w, b):
    """
    X: input data
    w: weights
    b: bias
    """
    z = np.dot(X, w) + b
    return sigmoid(z)

def accuracy(y, y_hat):
    y_pred = y_hat >= 0.5
    return np.mean(y == y_pred)

def test_cases():
    raise NotImplementedError()
    # Not sure how to test this yet

def plot(x, y, y_hat):
    plt.scatter(x, y)
    plt.plot(x, y_hat)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Binary Logistic Regression - Scratch')
    plt.show()


def main():
    # random data
    X = np.random.rand(100, 1).reshape(-1, 1)
    y = (X > 0.5).astype(int).reshape(-1)

    # Train
    w, b = fit(X, y)

    # Predict
    y_hat = predict(X, w, b)

    # Evaluate
    acc = accuracy(y, y_hat)
    print(f'Accuracy: {acc}')

    # Plot
    plot(X, y, y_hat)

if __name__ == '__main__':
    main()

