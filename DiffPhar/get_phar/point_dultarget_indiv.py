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


#
# # 读取文件并进行刚性配准
# vectors_1_coords_transformed, R, t = rigid_registration(vectors_1, vectors_2)

from rdkit import Geometry
from rdkit.Chem import rdMolTransforms
from rdkit.Numerics import rdAlignment
from rdkit.Chem.Pharm3D import EmbedLib


def align_pharmacophores(ref_pharma, probe_pharma):
    """ Align two pharmacophores against each other.

        Parameters
        ----------
        ref_pharma : np.ndarray
            Coordinates of the pharmacophoric points of the reference pharmacophore.
            Shape (n_points, 3)

        probe_pharma : np.ndarray
            Coordinates of the pharmacophoric points of probe pharmacophore.
            Shape (n_points, 3)

        Returns
        -------
        rmsd : float
            The root mean square deviation of the alignment.

        trans_mat : np.ndarray
            The transformation matrix. This matrix should be applied to the confomer of
            the probe_pharmacophore to obtain its updated positions.
    """
    ssd, trans_mat = rdAlignment.GetAlignmentTransform(ref_pharma, probe_pharma)
    rmsd = np.sqrt(ssd / ref_pharma.shape[0])
    return rmsd, trans_mat


center_1 = np.mean(vectors_1, axis=0)
center_2 = np.mean(vectors_2, axis=0)
translation_vector_ori = center_2 - center_1
vectors_1 = vectors_1 + translation_vector_ori


rmsd, trans_mat = align_pharmacophores(vectors_1, vectors_2)
# 分离旋转矩阵和平移向量
rotation_matrix = trans_mat[:3, :3]
translation_vector_new = trans_mat[:3, 3]

# 应用旋转矩阵并加上平移向量
transformed_vectors_1 = np.dot(vectors_1, rotation_matrix.T) + translation_vector_new



# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for vectors_1_coords_transformed (in red)
ax.scatter(transformed_vectors_1[:, 0], transformed_vectors_1[:, 1], transformed_vectors_1[:, 2], c='r', marker='o', label='PARP1')

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
vectors_1_coords_transformed_overlap, vectors_2_overlap,features_1_overlap, features_2_overlap = find_overlapped_cloud_and_features(transformed_vectors_1, vectors_2, features_1, features_2)


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

def cluster_and_visualize(vectors, features, n_clusters, output_filename):
    # Apply Gaussian Mixture Model (GMM) clustering
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm.fit(vectors)

    # Get cluster centers
    cluster_centers = gmm.means_

    # Initialize dictionaries for feature counts and probabilities
    feature_counts = {feature: np.zeros(n_clusters) for feature in features}
    feature_probs = {feature: np.zeros(n_clusters) for feature in features}
    max_prob_feature = {i: None for i in range(n_clusters)}

    # Count the occurrences and probabilities of each cluster for each feature
    for i in range(len(features)):
        feature = features[i]
        coordinates = vectors[i]

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
        ax.scatter(vectors[gmm.predict(vectors) == i, 0],
                   vectors[gmm.predict(vectors) == i, 1],
                   vectors[gmm.predict(vectors) == i, 2], label=f'Cluster {i + 1}')

    ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2], marker='x', s=100, c='red', label='Cluster Centers')

    ax.set_xlabel('X', weight='bold')
    ax.set_ylabel('Y', weight='bold')
    ax.set_zlabel('Z', weight='bold')

    # Set the viewing angle (elevation, azimuth)
    ax.view_init(elev=25, azim=135)  # You can adjust the values as needed

    # Move the legend slightly to the right
    ax.legend(loc='upper right', bbox_to_anchor=(0.15, 0.65), prop={'weight': 'bold'})

    # Open coordinate axis grid
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(visible=None, which='minor', axis='both', linestyle='', alpha=0)

    # Print feature frequencies and probabilities in each cluster
    cluster_info = []
    for i in range(n_clusters):
        print(f"\nCluster {i + 1}:")

        # Print cluster center coordinates
        print(f"Cluster Center Coordinates: {cluster_centers[i, 0], cluster_centers[i, 1], cluster_centers[i, 2]}")

        # Sort features by probability in descending order
        sorted_features = sorted(feature_probs.keys(), key=lambda x: feature_probs[x][i], reverse=True)

        print("Cluster Probabilities:")
        for feature in sorted_features:
            print(f"{feature}: {feature_probs[feature][i]}")

        most_common_feature = sorted_features[0]
        print("Most Common Feature:", most_common_feature)

        # Print feature frequencies and probabilities in each cluster
        cluster_info.append({
            'Cluster Center Coordinates': (cluster_centers[i, 0], cluster_centers[i, 1], cluster_centers[i, 2]),
            'Cluster Probabilities': {feature: feature_probs[feature][i] for feature in features},
            'Most Common Feature': max_prob_feature[i]
        })

    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()

    return cluster_info

# Apply the function to the first set of vectors and features
cluster_info_set1 = cluster_and_visualize(vectors_1_coords_transformed_overlap, features_1_overlap, 7,
                                          'dul_set1.png')

# Apply the function to the second set of vectors and features
cluster_info_set2 = cluster_and_visualize(vectors_2_overlap, features_2_overlap, 7, 'dul_set2.png')

def merge_clusters(cluster_info_set1, cluster_info_set2, threshold_set2, threshold_merge):
    merged_clusters = []

    # Helper function to check if a cluster is already merged
    def is_merged(cluster, merged_clusters):
        for merged_info in merged_clusters:
            distance = np.linalg.norm(np.array(cluster['Cluster Center Coordinates']) - np.array(merged_info['Cluster Center Coordinates']))
            if distance < threshold_merge:
                return True
        return False

    # Merge set1 clusters with set2 clusters
    for info1 in cluster_info_set1:
        min_distance = float('inf')
        closest_info2 = None

        for info2 in cluster_info_set2:
            distance = np.linalg.norm(np.array(info1['Cluster Center Coordinates']) - np.array(info2['Cluster Center Coordinates']))

            if distance < min_distance:
                min_distance = distance
                closest_info2 = info2

        # Check conditions for merging
        if min_distance < threshold_set2 and (closest_info2['Most Common Feature'] in ['LumpedHydrophobe', 'Aromatic'] or min_distance < threshold_merge):
            # Merge the clusters
            merged_info = {
                'Cluster Center Coordinates': [(c1 + c2) / 2 for c1, c2 in zip(info1['Cluster Center Coordinates'], closest_info2['Cluster Center Coordinates'])],
                'Cluster Probabilities': {},
                'Most Common Feature': max(info1['Most Common Feature'], closest_info2['Most Common Feature'], key=lambda x: info1['Cluster Probabilities'].get(x, 0) + closest_info2['Cluster Probabilities'].get(x, 0))
            }

            # Use maximum probability for common features
            common_features = set(info1['Cluster Probabilities']).intersection(closest_info2['Cluster Probabilities'])
            for feature in common_features:
                merged_info['Cluster Probabilities'][feature] = max(info1['Cluster Probabilities'].get(feature, 0), closest_info2['Cluster Probabilities'].get(feature, 0))

            # Add probabilities for unique features
            unique_features1 = set(info1['Cluster Probabilities']) - common_features
            for feature in unique_features1:
                merged_info['Cluster Probabilities'][feature] = info1['Cluster Probabilities'][feature]

            # unique_features2 = set(closest_info2['Cluster Probabilities']) - common_features
            # for feature in unique_features2:
            #     merged_info['Cluster Probabilities'][feature] = closest_info2['Cluster Probabilities'][feature]

            merged_clusters.append(merged_info)

    # Add set2 clusters that were not merged
    for info2 in cluster_info_set2:
        if not is_merged(info2, merged_clusters):
            merged_clusters.append(info2)

    # Add set1 clusters that were not merged
    for info1 in cluster_info_set1:
        if not is_merged(info1, merged_clusters):
            merged_clusters.append(info1)

    return merged_clusters


# Define the thresholds
threshold_set2 = 4  # Threshold for merging with set2 clusters
threshold_merge = 1  # General merge threshold

# Merge clusters
merged_clusters = merge_clusters(cluster_info_set1, cluster_info_set2, threshold_set2, threshold_merge)

# Print or use the merged cluster information as needed
for i, merged_info in enumerate(merged_clusters, start=1):
    print(f"\nMerged Cluster {i}:")
    print("Cluster Center Coordinates:", merged_info['Cluster Center Coordinates'])
    print("Cluster Probabilities:", merged_info['Cluster Probabilities'])

    # Find the feature with maximum probability
    max_prob_feature = max(merged_info['Cluster Probabilities'], key=merged_info['Cluster Probabilities'].get)
    print("Most Common Feature:", max_prob_feature)


# Visualize the cluster centers for both sets
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot Set 1 cluster centers
for i, info in enumerate(cluster_info_set1, start=1):
    coordinates = info['Cluster Center Coordinates']
    ax.scatter(coordinates[0], coordinates[1], coordinates[2], marker='x', s=150, c='red')

# Plot Set 2 cluster centers
for i, info in enumerate(cluster_info_set2, start=1):
    coordinates = info['Cluster Center Coordinates']
    ax.scatter(coordinates[0], coordinates[1], coordinates[2], marker='x', s=150, c='blue')

for i, info in enumerate(merged_clusters, start=1):
    coordinates = info['Cluster Center Coordinates']
    ax.scatter(coordinates[0], coordinates[1], coordinates[2], marker='x', s=25, c='green')


ax.set_xlabel('X', weight='bold')
ax.set_ylabel('Y', weight='bold')
ax.set_zlabel('Z', weight='bold')

# Set the viewing angle (elevation, azimuth)
ax.view_init(elev=25, azim=135)  # You can adjust the values as needed

# Move the legend slightly to the right
# ax.legend(loc='upper right', bbox_to_anchor=(0.15, 0.65), prop={'weight': 'bold'})

# Open coordinate axis grid
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(visible=None, which='minor', axis='both', linestyle='', alpha=0)

plt.show()










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

output_data = []

for merged_info in merged_clusters:
    # Use the merged_info data
    feature = max(merged_info['Cluster Probabilities'], key=merged_info['Cluster Probabilities'].get)
    if feature in mapping:
        phar_idx = mapping[feature]  # Use the mapping directly
        phar_type = idx2phar[phar_idx]
        coords = merged_info['Cluster Center Coordinates']
        output_data.append(f"{phar_type} {coords[0]:.2f} {coords[1]:.2f} {coords[2]:.2f}")

# Writing to the .posp file
with open("output_dual_indiv.posp", "w") as file:
    for line in output_data:
        file.write(line + "\n")



# output_data_1 = []
# output_data_2 = []
#
# for cluster, data in cluster_data.items():
#     feature = data['Most Probable Feature']
#     if feature in mapping:
#         phar_idx = mapping[feature]  # Use the mapping directly
#         phar_type = idx2phar[phar_idx]
#         coords = data['Center Coordinates']
#
#         # Save to the first file
#         output_data_1.append(f"{phar_type} {coords[0]:.2f} {coords[1]:.2f} {coords[2]:.2f}")
#
#         # Inverse transform coordinates to the original vector 1's coordinate space
#         coords_inverse_transformed = inverse_transform(coords, R, t)
#         coords_inverse_transformed_list = coords_inverse_transformed.tolist()
#         output_data_2.append(f"{phar_type} {' '.join(map(lambda x: f'{x:.2f}', coords_inverse_transformed_list))}")
#
# # Writing to the first .posp file
# with open("output_dual_1.posp", "w") as file:
#     for line in output_data_1:
#         file.write(line + "\n")
#
# # Writing to the second .posp file
# with open("output_dual_2.posp", "w") as file:
#     for line in output_data_2:
#         file.write(line + "\n")

