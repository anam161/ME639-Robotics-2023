import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import KDTree

# Define the lengths of the phalanges
L1 = 5  # length of first phalange
PIP = 7  # combined length of second and third phalanges (Link 2 + Link 3)
L4 = 2  # length of fourth phalange

# Define the lengths of the thumb phalanges
L1_thumb = 4  # length of first phalange (MCP to PIP)
L2_thumb = 3  # length of second phalange (PIP to DIP)
L3_thumb = 2  # length of third phalange (DIP to tip)

# Define the palm
palm_length = 7.5
palm_width = 7
palm_thickness = 2
palm_center = np.array([-3.2, 0.5, 3])

# Define the range of motion for each joint of the fingers
theta1_range = np.linspace(0, np.deg2rad(120), 10)  # Joint 1: Flexion/Extension
theta2_range = np.linspace(np.deg2rad(-10), np.deg2rad(10), 10)  # Joint 2: Abduction/Adduction
theta3_range = np.linspace(0, np.deg2rad(90), 10)  # Joint 3: Flexion/Extension
theta4_range = np.linspace(0, np.deg2rad(80), 10)  # Joint 4: Flexion/Extension

# Define the range of motion for each joint of the thumb
theta1_thumb_range = np.linspace(0, np.deg2rad(90), 10)  # MCP Joint: Flexion/Extension
theta2_thumb_range = np.linspace(np.deg2rad(-60), np.deg2rad(60), 10)  # MCP Joint: Rotation
theta3_thumb_range = np.linspace(0, np.deg2rad(90), 10)  # PIP Joint: Flexion/Extension
theta4_thumb_range = np.linspace(0, np.deg2rad(90), 10)  # DIP Joint: Flexion/Extension

# Create a meshgrid of all possible joint angles for the fingers
theta1, theta2, theta3, theta4 = np.meshgrid(theta1_range, theta2_range, theta3_range, theta4_range)

# Create a meshgrid of all possible joint angles for the thumb
theta1_thumb, theta2_thumb, theta3_thumb, theta4_thumb = np.meshgrid(
    theta1_thumb_range, theta2_thumb_range, theta3_thumb_range, theta4_thumb_range
)

# Function to compute the Cartesian coordinates for the given joint angles
def compute_positions(theta1, theta2, theta3, theta4, L1, L2, L3):
    x = L1 * np.cos(theta1) * np.cos(theta2) + L2 * np.cos(theta1 + theta3) * np.cos(theta2) + L3 * np.cos(theta1 + theta3 + theta4) * np.cos(theta2)
    y = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta3) + L3 * np.sin(theta1 + theta3 + theta4)
    z = L1 * np.cos(theta1) * np.sin(theta2) + L2 * np.cos(theta1 + theta3) * np.sin(theta2) + L3 * np.cos(theta1 + theta3 + theta4) * np.sin(theta2)
    return x, y, z

# Compute the positions for each segment of all fingers
x_thumb, y_thumb, z_thumb = compute_positions(theta1_thumb, theta2_thumb, theta3_thumb, theta4_thumb, L1_thumb, L2_thumb, L3_thumb)
x_index, y_index, z_index = compute_positions(theta1, theta2, theta3, theta4, L1, PIP, L4)
x_middle, y_middle, z_middle = compute_positions(theta1, theta2, theta3, theta4, L1, PIP, L4)
x_ring, y_ring, z_ring = compute_positions(theta1, theta2, theta3, theta4, L1, PIP, L4)

# Adjust thumb to be perpendicular to other fingers
def adjust_thumb_position(x, y, z):
    return z, y, x  # Swap x and z coordinates to simulate perpendicular thumb position

x_thumb, y_thumb, z_thumb = adjust_thumb_position(x_thumb, y_thumb, z_thumb)

# Flatten the arrays for further processing
def flatten_positions(x, y, z):
    return np.vstack((x.flatten(), y.flatten(), z.flatten())).T

points_thumb = flatten_positions(x_thumb, y_thumb, z_thumb)
points_index = flatten_positions(x_index, y_index, z_index)
points_middle = flatten_positions(x_middle, y_middle, z_middle)
points_ring = flatten_positions(x_ring, y_ring, z_ring)

# Generate points for the palm
def generate_palm_points(center, length, width, thickness, num_points=1000):
    x = np.random.uniform(center[0] - width / 2, center[0] + width / 2, num_points)
    y = np.random.uniform(center[1] - length / 2, center[1] + length / 2, num_points)
    z = np.random.uniform(center[2] - thickness / 2, center[2] + thickness / 2, num_points)
    return np.vstack((x, y, z)).T

points_palm = generate_palm_points(palm_center, palm_length, palm_width, palm_thickness)

# Combine the point clouds to form a complete set of points for collision detection
points_combined = [points_thumb, points_index, points_middle, points_ring, points_palm]

# Create labels for each segment including palm
segments = ["MCP", "PIP", "DIP"]
fingers = ["F1", "F2", "F3", "F4"]
labels = [f"{finger}{segment}" for finger in fingers for segment in segments] + ["PALM"]

# Function to compute self-collision map
def compute_self_collision_map(points_list, distance_threshold=1.0):
    num_segments = len(points_list) * len(segments) + 1
    collision_counts = np.zeros((num_segments, num_segments))
    
    for i in range(len(points_list)):
        for j in range(len(segments)):
            kdtree_i = KDTree(points_list[i])  # No noise added
            for k in range(i, len(points_list)):
                for l in range(len(segments)):
                    if i == k and j == l:
                        continue  # Skip self-collision within the same segment
                    kdtree_j = KDTree(points_list[k])
                    collision_pairs = kdtree_i.query_ball_tree(kdtree_j, r=distance_threshold)
                    count = sum(len(p) for p in collision_pairs)
                    collision_counts[i * len(segments) + j, k * len(segments) + l] = count
                    collision_counts[k * len(segments) + l, i * len(segments) + j] = count

    # Check collisions between palm and other segments
    kdtree_palm = KDTree(points_list[-1])
    for i in range(len(points_list) - 1):
        for j in range(len(segments)):
            kdtree_segment = KDTree(points_list[i])
            collision_pairs = kdtree_segment.query_ball_tree(kdtree_palm, r=distance_threshold)
            count = sum(len(p) for p in collision_pairs)
            collision_counts[i * len(segments) + j, -1] = count
            collision_counts[-1, i * len(segments) + j] = count
    
    return collision_counts

# Compute the self-collision map
collision_counts = compute_self_collision_map(points_combined)

# Normalize the values to be between 0 and 10
collision_counts_normalized = 10 * (collision_counts / np.max(collision_counts))

# Remove the extra row and column
collision_counts_normalized = collision_counts_normalized[:len(labels), :len(labels)]

# Create a heatmap to visualize the self-collision map
plt.figure(figsize=(12, 10))
sns.heatmap(collision_counts_normalized, annot=True, fmt=".1f", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Self-Collision Map in Cartesian Space")
plt.xlabel("Hand Link")
plt.ylabel("Hand Link")
plt.show()
