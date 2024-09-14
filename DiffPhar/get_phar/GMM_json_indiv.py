import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GaussianMixture
from matplotlib import rc

# 设置字体为新罗马字体
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})

# 读取保存的数据
input_file_path = 'phar_to_coords_no_tensor_usp1.json'  # 请根据实际路径修改
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

# Apply Gaussian Mixture Model (GMM) clustering
gmm = GaussianMixture(n_components=n_clusters, random_state=42)
gmm.fit(X)

# Get cluster centers
cluster_centers = gmm.means_

# Initialize feature_counts and feature_probs dictionaries
feature_counts = {feature: np.zeros(n_clusters) for molecule_features in loaded_phar_to_coords_no_tensor.values() for feature in molecule_features}
feature_probs = {feature: np.zeros(n_clusters) for molecule_features in loaded_phar_to_coords_no_tensor.values() for feature in molecule_features}

# Count the occurrences and probabilities of each cluster for each feature
for molecule, features in loaded_phar_to_coords_no_tensor.items():
    for feature, coordinates in features.items():
        labels = gmm.predict(coordinates)
        feature_counts[feature] += np.bincount(labels, minlength=n_clusters)
        probs = np.sum(gmm.predict_proba(coordinates), axis=0)
        feature_probs[feature] += probs

# Normalize the probabilities
for feature in feature_probs:
    feature_probs[feature] /= np.sum(feature_probs[feature])

# Visualize the clusters and annotate with the most probable feature
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(n_clusters):
    ax.scatter(X[gmm.predict(X) == i, 0], X[gmm.predict(X) == i, 1], X[gmm.predict(X) == i, 2], label=f'Cluster {i + 1}')

# Scatter plot for cluster centers
ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2], marker='x', s=100, c='red', label='Cluster Centers')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

# 打开坐标轴网格
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(visible=None, which='minor', axis='both', linestyle='', alpha=0)

# Print feature frequencies and probabilities in each cluster
for i in range(n_clusters):
    print(f"\nCluster {i + 1}:")

    # Print cluster center coordinates
    print(f"Cluster Center Coordinates: {cluster_centers[i, 0], cluster_centers[i, 1], cluster_centers[i, 2]}")

    # Calculate feature probabilities for the current cluster
    total_probs = 0
    total_frequencies = np.zeros(n_clusters)
    for molecule, coordinates in loaded_phar_to_coords_no_tensor.items():
        for feature, coords in coordinates.items():
            labels = gmm.predict(coords)
            total_frequencies += np.bincount(labels, minlength=n_clusters)
            total_probs += np.sum(gmm.predict_proba(coords), axis=0)

    feature_probs_dict = {feature: total_probs[j] / np.sum(total_probs) if np.sum(total_probs) > 0 else 0 for j, feature in enumerate(feature_probs)}

    print("Cluster Probabilities:")
    for feature, prob in feature_probs_dict.items():
        print(f"{feature}: {prob}")

    most_common_feature = max(feature_probs_dict, key=feature_probs_dict.get)
    print("Most Common Feature:", most_common_feature)

plt.show()
