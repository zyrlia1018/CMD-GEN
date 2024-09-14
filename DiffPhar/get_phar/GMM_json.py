import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GaussianMixture
from matplotlib import rc

# 设置字体为新罗马字体
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})

# 读取保存的数据
input_file_path = 'phar_to_coords_no_tensor_PARP1.json'  # 请根据实际路径修改
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
n_clusters = 7 # 5,6,7

# Apply Gaussian Mixture Model (GMM) clustering
gmm = GaussianMixture(n_components=n_clusters, random_state=42)
gmm.fit(X)

# Get cluster centers
cluster_centers = gmm.means_

# Initialize feature_counts and feature_probs dictionaries
feature_counts = {feature: np.zeros(n_clusters) for molecule_features in loaded_phar_to_coords_no_tensor.values() for feature in molecule_features}
feature_probs = {feature: np.zeros(n_clusters) for molecule_features in loaded_phar_to_coords_no_tensor.values() for feature in molecule_features}
max_prob_feature = {i: None for i in range(n_clusters)}

# Count the occurrences and probabilities of each cluster for each feature
for molecule, features in loaded_phar_to_coords_no_tensor.items():
    for feature, coordinates in features.items():
        labels = gmm.predict(coordinates)
        feature_counts[feature] += np.bincount(labels, minlength=n_clusters)
        probs = np.sum(gmm.predict_proba(coordinates), axis=0)
        feature_probs[feature] += probs
        max_prob_feature_for_molecule = np.argmax(probs)
        if max_prob_feature[labels[0]] is None or probs[max_prob_feature_for_molecule] > feature_probs[max_prob_feature[labels[0]]][labels[0]]:
            max_prob_feature[labels[0]] = feature

# Normalize the probabilities
for feature in feature_probs:
    feature_probs[feature] /= np.sum(feature_probs[feature])

# Visualize the clusters and annotate with the most probable feature
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(n_clusters):
    ax.scatter(X[gmm.predict(X) == i, 0], X[gmm.predict(X) == i, 1], X[gmm.predict(X) == i, 2], label=f'Cluster {i + 1}')

ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2], marker='x', s=100, c='red', label='Cluster Centers')

ax.set_xlabel('X', weight='bold')
ax.set_ylabel('Y', weight='bold')
ax.set_zlabel('Z', weight='bold')

# Set the viewing angle (elevation, azimuth)
ax.view_init(elev=20, azim=135)  # You can adjust the values as needed

# Move the legend slightly to the right
ax.legend(loc='upper right', bbox_to_anchor=(0.15, 0.65), prop={'weight': 'bold'})


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
    # Sort features by probability in descending order
    sorted_features = sorted(feature_probs.keys(), key=lambda x: feature_probs[x][i], reverse=True)

    print("Cluster Probabilities:")
    for feature in sorted_features:
        print(f"{feature}: {feature_probs[feature][i]}")

        print("Feature Frequencies:", feature_counts[feature])
    most_common_feature = sorted_features[0]
    # ax.text(cluster_centers[i, 0], cluster_centers[i, 1], cluster_centers[i, 2], f'{most_common_feature}', fontsize=8, ha='left', va='bottom')

    print("Most Common Feature:", most_common_feature)

plt.show()



# Print cluster center coordinates and most common feature for each cluster
# Create a dictionary to store cluster data
cluster_data = {}

# Print feature frequencies and probabilities in each cluster
for i in range(n_clusters):
    # Find the most common feature for the current cluster
    most_common_feature_for_cluster = sorted(feature_probs.keys(), key=lambda x: feature_probs[x][i], reverse=True)

    # Add cluster data to the dictionary
    cluster_data[f'Cluster {i + 1}'] = {
        'Center Coordinates': (cluster_centers[i, 0], cluster_centers[i, 1], cluster_centers[i, 2]),
        'Most Probable Feature': most_common_feature_for_cluster[0]
    }

# Print the dictionary
print(cluster_data)

mapping = {
    'Aromatic': 1,
    'Hydrophobe': 2,
    'PosIonizable': 3,
    'Acceptor': 4,
    'Donor': 5,
    'LumpedHydrophobe': 6,
    'others': 7,
}

idx2phar = {
    1: 'AROM',
    2: 'HYBL',
    3: 'POSC',
    4: 'HACC',
    5: 'HDON',
    6: 'LHYBL',
    7: 'UNKNOWN'
}

output_data = []

for cluster, data in cluster_data.items():
    feature = data['Most Probable Feature']
    if feature in mapping:
        phar_idx = mapping[feature]  # Use the mapping directly
        phar_type = idx2phar[phar_idx]
        coords = data['Center Coordinates']
        output_data.append(f"{phar_type} {coords[0]:.2f} {coords[1]:.2f} {coords[2]:.2f}")

# Writing to the .posp file
with open("output.posp", "w") as file:
    for line in output_data:
        file.write(line + "\n")