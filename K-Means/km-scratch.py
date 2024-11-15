"""
K-Means from scratch

Steps:
1. Randomly initialize K cluster centroids
2. Assign each data point to the nearest cluster centroid
3. Compute the new cluster centroids by taking the average of all the data points assigned to each cluster

if the cluster centroids do not change, then the algorithm has converged, otherwise repeat steps 2 and 3

Author: Alex Javadi <alex@aljm.org>
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs # Random data

def distance(data, centroids):
    """
    Euclidean distance between two points
    
    F: Distance(data, centroids) sqrt(sum((data - centroids) ** 2))
    """
    return np.sqrt(np.sum((data - centroids) ** 2, axis=1))

def wcss(data, centroids, labels):
    """
    The objective of K-Means is to minimize the within-cluster sum of squares (WCSS)
    
    F = WCSS(data, centroids, labels) = sum(sum((data[labels == i] - centroids[i]) ** 2))
    """
    return np.sum([np.sum((data[labels == i] - centroids[i]) ** 2) for i in range(centroids.shape[0])])


def kmeans(data, k, max_iter=100):
    """
    K-Means algorithm
    
    F: KMeans(data, k, max_iter) -> (centroids, labels)
    """

    # K cluster centroids
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    
    for _ in range(max_iter):
        # Assign each data point to the nearest cluster centroid
        labels = np.argmin(np.array([distance(data, centroid) for centroid in centroids]), axis=0)
        
        # Compute the new cluster centroids by taking the average of all the data points assigned to each cluster
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        
        # If the cluster centroids do not change, then the algorithm has converged
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    return centroids, labels

def test():
    """
    Test K-Means

    Since K-Means is an unsupervised learning algorithm, we can only test the output of the algorithm (As I googled, there is no way to test the output of K-Means)
    """
    k = 3
    data, _ = make_blobs(n_samples=1000, centers=3, n_features=2, random_state=42)

    assert kmeans(data, k)[0].shape == (k, 2)
    assert kmeans(data, k)[1].shape == (1000,)
    assert wcss(data, kmeans(data, k)[0], kmeans(data, k)[1]) > 0

if __name__ == '__main__':


    # Generate random data
    data, _ = make_blobs(n_samples=1000, centers=3, n_features=2, random_state=42)
    
    # K-Means
    k = 3
    centroids, labels = kmeans(data, k)
    
    # Plot
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')
    plt.title('K-Means from scratch')
    plt.show()
    
    # Within-cluster sum of squares (WCSS)
    print(wcss(data, centroids, labels))