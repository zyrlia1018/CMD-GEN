import json
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D

# 设置字体为新罗马字体
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})

mapping = {
    'Aromatic': 1,
    'Hydrophobe': 2,
    'PosIonizable': 3,
    'Acceptor': 4,
    'Donor': 5,
    'LumpedHydrophobe': 6,
    'others': 7,
}


# 读取第一个文件
input_file_path_1 = 'phar_to_coords_no_tensor_PARP1.json'
with open(input_file_path_1, 'r') as f:
    data_1 = json.load(f)

# 读取第二个文件
input_file_path_2 = 'phar_to_coords_no_tensor_PI3K.json'
with open(input_file_path_2, 'r') as f:
    data_2 = json.load(f)

# Convert data to vectors
vectors_1 = []
features_1 = []
for molecule, features in data_1.items():
    for feature, coordinates in features.items():
        vectors_1.extend(coordinates)
        features_1.extend([feature] * len(coordinates))

# Convert data to vectors
vectors_2 = []
features_2 = []
for molecule, features in data_2.items():
    for feature, coordinates in features.items():
        vectors_2.extend(coordinates)
        features_2.extend([feature] * len(coordinates))

vectors_1 = np.array(vectors_1)
vectors_2 = np.array(vectors_2)

def rigid_transform_Kabsch_3D(A, B):
    assert A.shape[1] == B.shape[1]
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise: 3 x 1
    centroid_A = np.mean(A, axis=1, keepdims=True)
    centroid_B = np.mean(B, axis=1, keepdims=True)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ Bm.T

    # find rotation
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        SS = np.diag([1., 1., -1.])
        R = (Vt.T @ SS) @ U.T
    assert np.isclose(np.linalg.det(R), 1.0)

    t = -R @ centroid_A + centroid_B
    return R, t

def rigid_registration(vectors_1, vectors_2):
    R, t = rigid_transform_Kabsch_3D(vectors_1.T, vectors_2.T)
    coords_transformed = (R @ (vectors_1).T).T + t.squeeze()
    return coords_transformed, R, t

def inverse_transform(coords, R, t):
    # Convert the tuple to a NumPy array
    coords_array = np.array(coords)

    # Reshape to a column vector
    coords_column_vector = coords_array.reshape(-1, 1)

    # Apply the inverse transformation
    coords_transformed = (np.linalg.inv(R) @ (coords_column_vector - t)).squeeze()

    return coords_transformed



# 读取文件并进行刚性配准
vectors_1_coords_transformed, R, t = rigid_registration(vectors_1, vectors_2)



# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for vectors_1_coords_transformed (in red)
ax.scatter(vectors_1_coords_transformed[:, 0], vectors_1_coords_transformed[:, 1], vectors_1_coords_transformed[:, 2], c='r', marker='o', label='PARP1')

# Scatter plot for vectors_2 (in blue)
ax.scatter(vectors_2[:, 0], vectors_2[:, 1], vectors_2[:, 2], c='b', marker='o', label='PI3K')


ax.set_xlabel('X', weight='bold')
ax.set_ylabel('Y', weight='bold')
ax.set_zlabel('Z', weight='bold')

# Set the viewing angle (elevation, azimuth)
ax.view_init(elev=25, azim=135)  # You can adjust the values as needed

# Move the legend slightly to the right
ax.legend(loc='upper right', bbox_to_anchor=(0.15, 0.65), prop={'weight': 'bold'})

# 打开坐标轴网格
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(visible=None, which='minor', axis='both', linestyle='', alpha=0)

# plt.legend()
plt.savefig('dul_all.png', dpi=300, bbox_inches='tight')
plt.show()

import open3d as o3d
import numpy as np


def find_overlapped_cloud_and_features(vectors_1, vectors_2, features_1, features_2, threshold=1.5):
    # Convert Python lists to NumPy arrays
    vectors_1 = np.array(vectors_1)
    vectors_2 = np.array(vectors_2)

    # Create Open3D point clouds
    cloud1 = o3d.geometry.PointCloud()
    cloud1.points = o3d.utility.Vector3dVector(vectors_1)

    cloud2 = o3d.geometry.PointCloud()
    cloud2.points = o3d.utility.Vector3dVector(vectors_2)

    dists1 = cloud1.compute_point_cloud_distance(cloud2)
    dists2 = cloud2.compute_point_cloud_distance(cloud1)

    dists1 = np.asarray(dists1)
    dists2 = np.asarray(dists2)

    ind1 = np.where(dists1 < threshold)[0]
    ind2 = np.where(dists2 < threshold)[0]

    overlapped_cloud1 = cloud1.select_by_index(ind1)
    overlapped_cloud2 = cloud2.select_by_index(ind2)

    # Extract coordinates and features
    vectors_1_overlap = np.asarray(overlapped_cloud1.points)
    vectors_2_overlap = np.asarray(overlapped_cloud2.points)

    # Extract features for the overlapped points using index
    features_1_overlap = [features_1[i] for i in ind1]
    features_2_overlap = [features_2[i] for i in ind2]

    return vectors_1_overlap, vectors_2_overlap, features_1_overlap, features_2_overlap


# Example usage:
vectors_1_coords_transformed_overlap, vectors_2_overlap,features_1_overlap, features_2_overlap = find_overlapped_cloud_and_features(vectors_1_coords_transformed, vectors_2, features_1, features_2)


# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for vectors_1_coords_transformed (in red)
ax.scatter(vectors_1_coords_transformed_overlap[:, 0], vectors_1_coords_transformed_overlap[:, 1], vectors_1_coords_transformed_overlap[:, 2], c='r', marker='o', label='PARP1')
ax.scatter(vectors_2_overlap[:, 0], vectors_2_overlap[:, 1], vectors_2_overlap[:, 2], c='b', marker='o', label='PI3K')

ax.set_xlabel('X', weight='bold')
ax.set_ylabel('Y', weight='bold')
ax.set_zlabel('Z', weight='bold')

# Set the viewing angle (elevation, azimuth)
ax.view_init(elev=25, azim=135)  # You can adjust the values as needed

# Move the legend slightly to the right
ax.legend(loc='upper right', bbox_to_anchor=(0.15, 0.65), prop={'weight': 'bold'})

# 打开坐标轴网格
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(visible=None, which='minor', axis='both', linestyle='', alpha=0)

# plt.legend()
plt.savefig('dul_all_overlap.png', dpi=300, bbox_inches='tight')
plt.show()







from sklearn.mixture import GaussianMixture
# 合并两组点云的坐标和特征
merged_vectors = np.concatenate((vectors_1_coords_transformed_overlap, vectors_2_overlap), axis=0)
merged_features = features_1_overlap + features_2_overlap  # 合并特征信息

# Choose the number of clusters (you may need to tune this)
n_clusters = 7 # 5,6,7

# Apply Gaussian Mixture Model (GMM) clustering
gmm = GaussianMixture(n_components=n_clusters, random_state=42)
gmm.fit(merged_vectors)

# Get cluster centers
cluster_centers = gmm.means_

# Initialize feature_counts and feature_probs dictionaries
feature_counts = {feature: np.zeros(n_clusters) for feature in merged_features}
feature_probs = {feature: np.zeros(n_clusters) for feature in merged_features}
max_prob_feature = {i: None for i in range(n_clusters)}

# Count the occurrences and probabilities of each cluster for each feature
for i in range(len(merged_features)):
    feature = merged_features[i]
    coordinates = merged_vectors[i]

    labels = gmm.predict(coordinates.reshape(1, -1))
    feature_counts[feature] += np.bincount(labels, minlength=n_clusters)
    probs = gmm.predict_proba(coordinates.reshape(1, -1))[0]
    feature_probs[feature] += probs
    max_prob_feature_for_sample = np.argmax(probs)

    if max_prob_feature[labels[0]] is None or probs[max_prob_feature_for_sample] > feature_probs[max_prob_feature[labels[0]]][labels[0]]:
        max_prob_feature[labels[0]] = feature

# Normalize the probabilities
for feature in feature_probs:
    feature_probs[feature] /= np.sum(feature_probs[feature])

# Visualize the clusters and annotate with the most probable feature
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(n_clusters):
    ax.scatter(merged_vectors[gmm.predict(merged_vectors) == i, 0], merged_vectors[gmm.predict(merged_vectors) == i, 1],
               merged_vectors[gmm.predict(merged_vectors) == i, 2], label=f'Cluster {i + 1}')

ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2], marker='x', s=100, c='red', label='Cluster Centers')

ax.set_xlabel('X', weight='bold')
ax.set_ylabel('Y', weight='bold')
ax.set_zlabel('Z', weight='bold')

# Set the viewing angle (elevation, azimuth)
ax.view_init(elev=25, azim=135)  # You can adjust the values as needed

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

plt.savefig('dul_all.png', dpi=300, bbox_inches='tight')
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
    'NegIonizable': 7,
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

# output_data = []
#
# for cluster, data in cluster_data.items():
#     feature = data['Most Probable Feature']
#     if feature in mapping:
#         phar_idx = mapping[feature]  # Use the mapping directly
#         phar_type = idx2phar[phar_idx]
#         coords = data['Center Coordinates']
#         output_data.append(f"{phar_type} {coords[0]:.2f} {coords[1]:.2f} {coords[2]:.2f}")

# # Writing to the .posp file
# with open("output_dual.posp", "w") as file:
#     for line in output_data:
#         file.write(line + "\n")




output_data_1 = []
output_data_2 = []

for cluster, data in cluster_data.items():
    feature = data['Most Probable Feature']
    if feature in mapping:
        phar_idx = mapping[feature]  # Use the mapping directly
        phar_type = idx2phar[phar_idx]
        coords = data['Center Coordinates']

        # Save to the first file
        output_data_1.append(f"{phar_type} {coords[0]:.2f} {coords[1]:.2f} {coords[2]:.2f}")

        # Inverse transform coordinates to the original vector 1's coordinate space
        coords_inverse_transformed = inverse_transform(coords, R, t)
        coords_inverse_transformed_list = coords_inverse_transformed.tolist()
        output_data_2.append(f"{phar_type} {' '.join(map(lambda x: f'{x:.2f}', coords_inverse_transformed_list))}")

# Writing to the first .posp file
with open("output_dual_1.posp", "w") as file:
    for line in output_data_1:
        file.write(line + "\n")

# Writing to the second .posp file
with open("output_dual_2.posp", "w") as file:
    for line in output_data_2:
        file.write(line + "\n")

