import numpy as np
import matplotlib.pyplot as plt
import json
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from collections import Counter

# 读取保存的数据
input_file_path = 'phar_to_coords_no_tensor_PARP30.json'  # 请根据实际路径修改
with open(input_file_path, 'r') as f:
    loaded_phar_to_coords_no_tensor = json.load(f)

# Convert data to vectors
vectors = []
for molecule, features in loaded_phar_to_coords_no_tensor.items():
    for feature, coordinates in features.items():
        vectors.extend(coordinates)

# Convert to NumPy array
X = np.array(vectors)

# Choose the number of clusters (you may need to tune this)
n_clusters = 7

# Apply K-Means clustering
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(X)

# Get cluster centers
cluster_centers = kmeans.cluster_centers_

# Count the occurrences of each cluster for each feature
feature_counts = {feature: np.zeros(n_clusters) for molecule_features in loaded_phar_to_coords_no_tensor.values() for feature in molecule_features}
for molecule, features in loaded_phar_to_coords_no_tensor.items():
    for feature, coordinates in features.items():
        labels = kmeans.predict(coordinates)
        feature_counts[feature] += np.bincount(labels, minlength=n_clusters)

# Visualize the clusters
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(n_clusters):
    ax.scatter(X[kmeans.labels_ == i, 0], X[kmeans.labels_ == i, 1], X[kmeans.labels_ == i, 2], label=f'Cluster {i + 1}')

ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2], marker='x', s=200, c='red', label='Cluster Centers')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()

# Print feature frequencies in each cluster
for feature, counts in feature_counts.items():
    print(f"\nFeature: {feature}")
    print("Cluster Frequencies:", counts)
    most_common_cluster = np.argmax(counts)
    print("Most Common Cluster:", most_common_cluster + 1)  # Adding 1 to make it human-readable
