import json
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D

# 设置字体为新罗马字体
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})

# 读取第一个文件
input_file_path_1 = 'phar_to_coords_no_tensor_PARP1_select.json'
with open(input_file_path_1, 'r') as f:
    data_1 = json.load(f)

# 读取第二个文件
input_file_path_2 = 'phar_to_coords_no_tensor_PARP2_select.json'
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
for molecule, features in data_2.items():
    for feature, coordinates in features.items():
        vectors_2.extend(coordinates)

vectors_1 = np.array(vectors_1)
vectors_2 = np.array(vectors_2)
# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for vectors_1_coords_transformed (in red)
ax.scatter(vectors_1[:, 0], vectors_1[:, 1], vectors_1[:, 2], c='limegreen', marker='o', label='PARP1')

# Scatter plot for vectors_2 (in blue)
ax.scatter(vectors_2[:, 0], vectors_2[:, 1], vectors_2[:, 2], c='slateblue', marker='o', label='PARP2')


ax.set_xlabel('X', weight='bold')
ax.set_ylabel('Y', weight='bold')
ax.set_zlabel('Z', weight='bold')

# Set the viewing angle (elevation, azimuth)
ax.view_init(elev=25, azim=-45)  # You can adjust the values as needed

# Move the legend slightly to the right
ax.legend(loc='upper right', bbox_to_anchor=(0.15, 0.65), prop={'weight': 'bold'})

# 打开坐标轴网格
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(visible=None, which='minor', axis='both', linestyle='', alpha=0)

# plt.legend()
plt.savefig('select_all.png', dpi=300, bbox_inches='tight')
plt.show()

# Create Open3D point clouds from vectors
cloud_1 = o3d.geometry.PointCloud()
cloud_1.points = o3d.utility.Vector3dVector(np.array(vectors_1))

cloud_2 = o3d.geometry.PointCloud()
cloud_2.points = o3d.utility.Vector3dVector(np.array(vectors_2))

# Compute convex hull of the second point cloud
hull, _ = cloud_2.compute_convex_hull()

# Compute distances from each point in cloud_1 to the convex hull of cloud_2
distances = np.asarray(cloud_1.compute_point_cloud_distance(cloud_2))

# Get points from cloud_1 outside the convex hull by a distance greater than 1
indices_outside_hull = np.where(distances > 1)[0]
points_outside_hull = cloud_1.select_by_index(indices_outside_hull)

# Extract three-dimensional coordinates and features of points outside the convex hull
new_points = np.asarray(points_outside_hull.points)
new_features = [features_1[idx] for idx in indices_outside_hull]

import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Assuming you have already defined new_points and new_features
data = np.asarray(new_points)

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.8, min_samples=5)
labels = dbscan.fit_predict(data)

# Collect feature probabilities and cluster centers for each cluster
feature_probs_clusters = {label: {feature: 0.0 for feature in set(new_features)} for label in set(labels)}
cluster_centers = {}

# Count the occurrences and probabilities of each feature in each cluster
for label in set(labels):
    cluster_indices = np.where(labels == label)[0]
    cluster_features = [new_features[idx] for idx in cluster_indices]

    # Count occurrences of each feature in the cluster
    feature_counts = {feature: cluster_features.count(feature) for feature in set(cluster_features)}

    # Calculate probabilities
    total_points = len(cluster_features)
    for feature in feature_probs_clusters[label]:
        feature_probs_clusters[label][feature] = feature_counts.get(feature, 0) / total_points

    # Calculate cluster center
    cluster_centers[label] = np.mean(data[labels == label], axis=0)

# Print feature probabilities and cluster centers for each cluster
for label, feature_probs in feature_probs_clusters.items():
    print(f"\nCluster {label} Feature Probabilities:")
    for feature, prob in feature_probs.items():
        print(f"{feature}: {prob}")

    print(f"Cluster {label} Center Coordinates: {cluster_centers[label]}")
# Plot the clustered points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

unique_labels = np.unique(labels)
for label in unique_labels:
    if label == -1:
        # Plot outliers as black points
        outlier_points = data[labels == label]
        ax.scatter(outlier_points[:, 0], outlier_points[:, 1], outlier_points[:, 2], c='black', marker='o', s=50, label='Outliers')
    else:
        # Plot clustered points with different colors
        cluster_points = data[labels == label]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], label=f'Cluster {label + 1}', s=50)


ax.set_xlabel('X', weight='bold')
ax.set_ylabel('Y', weight='bold')
ax.set_zlabel('Z', weight='bold')

# Set the viewing angle (elevation, azimuth)
ax.view_init(elev=25, azim=-45)  # You can adjust the values as needed

# Move the legend slightly to the right
ax.legend(loc='upper right', bbox_to_anchor=(0.15, 0.65), prop={'weight': 'bold'})

# 打开坐标轴网格
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(visible=None, which='minor', axis='both', linestyle='', alpha=0)

# plt.legend()
plt.savefig('select_phar.png', dpi=300, bbox_inches='tight')
plt.show()


# # Create a new point cloud from extracted coordinates and features
# new_cloud = o3d.geometry.PointCloud()
# new_cloud.points = o3d.utility.Vector3dVector(new_points)
#
# # Print or use new_features as needed
# print("Extracted Features:", new_features)
#
# # Visualize the results
# o3d.visualization.draw_geometries([new_cloud])