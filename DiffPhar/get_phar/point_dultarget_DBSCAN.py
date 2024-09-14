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
input_file_path_1 = 'phar_to_coords_no_tensor_PARP1_select.json'
with open(input_file_path_1, 'r') as f:
    data_1 = json.load(f)

# 读取第二个文件
input_file_path_2 = 'phar_to_coords_no_tensor_PI3K_dul.json'
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
#
#




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

# # Scatter plot for vectors_1_coords_transformed (in red)
# ax.scatter(vectors_1_coords_transformed[:, 0], vectors_1_coords_transformed[:, 1], vectors_1_coords_transformed[:, 2], c='r', marker='o', label='PARP1')
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





from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

# Assuming X contains your spatial coordinates
X = np.concatenate((vectors_1_coords_transformed_overlap, vectors_2_overlap), axis=0)
merged_features = features_1_overlap + features_2_overlap  # Combine feature information

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.2, min_samples=12)
labels = dbscan.fit_predict(X_scaled)

# Print feature probabilities, most probable feature, and cluster centers for each cluster
feature_probs_dbscan = {feature: np.zeros(len(np.unique(labels))) for feature in merged_features}
max_prob_feature_dbscan = {label: None for label in np.unique(labels)}
cluster_centers_dbscan = {label: None for label in np.unique(labels)}

for i in range(len(merged_features)):
    feature = merged_features[i]
    coordinates = X_scaled[i]

    cluster_label = labels[i]

    if cluster_label != -1:
        feature_probs_dbscan[feature][cluster_label] += 1

        # Update most probable feature for the cluster
        if max_prob_feature_dbscan[cluster_label] is None or feature_probs_dbscan[feature][cluster_label] > feature_probs_dbscan[max_prob_feature_dbscan[cluster_label]][cluster_label]:
            max_prob_feature_dbscan[cluster_label] = feature

        # Update cluster center if not already set
        if cluster_centers_dbscan[cluster_label] is None:
            cluster_centers_dbscan[cluster_label] = coordinates

# Normalize the probabilities
for feature in feature_probs_dbscan:
    feature_probs_dbscan[feature] /= np.sum(feature_probs_dbscan[feature])

# Print feature probabilities, most probable feature, and cluster centers for each cluster
print("\nFeature Probabilities (DBSCAN):")
for label in np.unique(labels):
    if label != -1:
        print(f"\nCluster {label + 1}:")

        # Use a set to keep track of printed features
        printed_features = set()

        # Print non-zero feature probabilities for each cluster
        for i, feature in enumerate(merged_features):
            probability = feature_probs_dbscan[feature][label]
            if probability > 0 and feature not in printed_features:
                print(f"{feature} Probability:", probability)
                printed_features.add(feature)

        print("Most Probable Feature:", max_prob_feature_dbscan[label])
        print("Cluster Center Coordinates:", cluster_centers_dbscan[label])


# Visualize the clusters
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for each cluster
for label in np.unique(labels):
    if label == -1:
        # Noise points
        noise_mask = (labels == label)
        ax.scatter(X_scaled[noise_mask, 0], X_scaled[noise_mask, 1], X_scaled[noise_mask, 2], label='Noise', c='gray', alpha=0.5)
    else:
        cluster_mask = (labels == label)
        ax.scatter(X_scaled[cluster_mask, 0], X_scaled[cluster_mask, 1], X_scaled[cluster_mask, 2], label=f'Cluster {label + 1}')

ax.set_xlabel('X', weight='bold')
ax.set_ylabel('Y', weight='bold')
ax.set_zlabel('Z', weight='bold')

# Set the viewing angle (elevation, azimuth)
ax.view_init(elev=25, azim=135)  # You can adjust the values as needed

# Move the legend slightly to the right
ax.legend(loc='upper right', bbox_to_anchor=(0.15, 0.65), prop={'weight': 'bold'})

plt.show()
