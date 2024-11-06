"""
K-Nearest Neighbors (KNN) using scikit-learn


Author: Alex Javadi <alex@aljm.org>
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Generate synthetic data (I wasn't sure about the data generation part, so I used the same data as the scratch implementation)
X_class0 = np.random.normal(loc=0.5, scale=0.1, size=(50, 2))
y_class0 = np.zeros(50, dtype=int)

X_class1 = np.random.normal(loc=0.8, scale=0.1, size=(50, 2))
y_class1 = np.ones(50, dtype=int)

# Combine the data
X = np.vstack([X_class0, X_class1])
y = np.hstack([y_class0, y_class1])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
k = 5
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Plot the data
plt.figure(figsize=(8, 6))
plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], label='Class 0', alpha=0.6)
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], label='Class 1', alpha=0.6)

plt.xlabel('Feature 1 (X1)')
plt.ylabel('Feature 2 (X2)')
plt.title('K-Nearest Neighbours - scikit-learn')
plt.legend()
plt.grid(True)
plt.show()