"""
K-Means Clustering using sklearn

Author: Alex Javadi <alex@aljm.org>
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Suggested by GPT for testing
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score

# Random data
data, _ = make_blobs(n_samples=1000, centers=3, n_features=2, random_state=42)
k = 3

# K-Means algorithm
kmeans = KMeans(n_clusters=k)
kmeans.fit(data)

# Plot the data points and cluster centroids
plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, cmap='viridis')

centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, marker='x')

plt.title('K-Means using sklearn')
plt.show()

# Suggested by GPT for testing
silhouette = silhouette_score(data, kmeans.labels_)
calinski_harabasz = calinski_harabasz_score(data, kmeans.labels_)

print(f'Silhouette Score: {silhouette}')
print(f'Calinski-Harabasz Score: {calinski_harabasz}')
print(f'WCSS: {kmeans.inertia_}')  # Within-cluster sum of squares

