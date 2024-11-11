"""
Random Forest using sklearn

Author: Alex Javadi <alex@aljm.org>
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# Generate a random dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Plot
plt.figure(figsize=(10, 6))
plt.plot(range(len(y_test)), y_test, 'bo', markersize=5, label='True')
plt.plot(range(len(y_pred)), y_pred, 'ro', markersize=3, label='Predicted')
plt.legend()
plt.title('Random Forest Classifier - sklearn')
plt.xlabel('Sample index')
plt.ylabel('Class label')
plt.show()