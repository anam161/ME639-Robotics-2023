#!/home/ubuntu/Desktop/Anam/.venv/bin/python
import numpy as np
import pyvista as pv
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import seaborn as sns

# Define the lengths of the phalanges
L1 = 5  # length of first phalange
PIP = 7  # combined length of second and third phalanges (Link 2 + Link 3)
L4 = 2  # length of fourth phalange

# Define the lengths of the thumb phalanges
L1_thumb = 4  # length of first phalange (MCP to PIP)
L2_thumb = 3  # length of second phalange (PIP to DIP)
L3_thumb = 2  # length of third phalange (DIP to tip)

# Define the range of motion for each joint of the fingers
theta1_range = np.linspace(0, np.deg2rad(90), 50)  # Joint 1: Flexion/Extension
theta2_range = np.linspace(np.deg2rad(-30), np.deg2rad(30), 50)  # Joint 2: Abduction/Adduction
theta3_range = np.linspace(0, np.deg2rad(100), 50)  # Joint 3: Flexion/Extension
theta4_range = np.linspace(0, np.deg2rad(80), 50)  # Joint 4: Flexion/Extension

# Define the range of motion for each joint of the thumb
theta1_thumb_range = np.linspace(0, np.deg2rad(90), 50)  # MCP Joint: Flexion/Extension
theta2_thumb_range = np.linspace(np.deg2rad(-30), np.deg2rad(30), 50)  # MCP Joint: Rotation
theta3_thumb_range = np.linspace(0, np.deg2rad(90), 50)  # PIP Joint: Flexion/Extension
theta4_thumb_range = np.linspace(0, np.deg2rad(90), 50)  # DIP Joint: Flexion/Extension

# Create a meshgrid of all possible joint angles for the fingers
theta1, theta2, theta3, theta4 = np.meshgrid(theta1_range, theta2_range, theta3_range, theta4_range)

# Create a meshgrid of all possible joint angles for the thumb
theta1_thumb, theta2_thumb, theta3_thumb, theta4_thumb = np.meshgrid(
    theta1_thumb_range, theta2_thumb_range, theta3_thumb_range, theta4_thumb_range
)

# Calculate the Cartesian positions for the index finger's first link (Flexion/Extension)
x1_index = L1 * np.cos(theta1)
y1_index = L1 * np.sin(theta1)
z1_index = np.random.uniform(-1e-3, 1e-3, x1_index.shape)  # Small random perturbation

# Flatten the arrays for plotting
x1_index = x1_index.flatten()
y1_index = y1_index.flatten()
z1_index = z1_index.flatten()

# Calculate the Cartesian positions for the PIP link (Abduction/Adduction and Flexion/Extension)
x2_index = L1 * np.cos(theta1) * np.cos(theta2) + PIP * np.cos(theta1 + theta3) * np.cos(theta2)
y2_index = L1 * np.sin(theta1) + PIP * np.sin(theta1 + theta3)
z2_index = L1 * np.cos(theta1) * np.sin(theta2) + PIP * np.cos(theta1 + theta3) * np.sin(theta2)

# Flatten the arrays for plotting
x2_index = x2_index.flatten()
y2_index = y2_index.flatten()
z2_index = z2_index.flatten()

# Calculate the Cartesian positions for the fourth link
x4_index = (L1 * np.cos(theta1) * np.cos(theta2) + 
            PIP * np.cos(theta1 + theta3) * np.cos(theta2) + 
            L4 * np.cos(theta1 + theta3 + theta4) * np.cos(theta2))
y4_index = (L1 * np.sin(theta1) + 
            PIP * np.sin(theta1 + theta3) + 
            L4 * np.sin(theta1 + theta3 + theta4))
z4_index = (L1 * np.cos(theta1) * np.sin(theta2) + 
            PIP * np.cos(theta1 + theta3) * np.sin(theta2) + 
            L4 * np.cos(theta1 + theta3 + theta4) * np.sin(theta2))

# Flatten the arrays for plotting
x4_index = x4_index.flatten()
y4_index = y4_index.flatten()
z4_index = z4_index.flatten()

# Calculate the Cartesian positions for the middle finger's first link (Flexion/Extension)
x1_middle = L1 * np.cos(theta1)
y1_middle = L1 * np.sin(theta1)   # Offset by 10 units in the y direction
z1_middle = np.random.uniform(-1e-3, 1e-3, x1_middle.shape) + 2 # Small random perturbation

# Flatten the arrays for plotting
x1_middle = x1_middle.flatten()
y1_middle = y1_middle.flatten()
z1_middle = z1_middle.flatten()

# Calculate the Cartesian positions for the PIP link (Abduction/Adduction and Flexion/Extension)
x2_middle = L1 * np.cos(theta1) * np.cos(theta2) + PIP * np.cos(theta1 + theta3) * np.cos(theta2)
y2_middle = L1 * np.sin(theta1) + PIP * np.sin(theta1 + theta3) # Offset by 10 units in the y direction
z2_middle = L1 * np.cos(theta1) * np.sin(theta2) + PIP * np.cos(theta1 + theta3) * np.sin(theta2) + 2

# Flatten the arrays for plotting
x2_middle = x2_middle.flatten()
y2_middle = y2_middle.flatten()
z2_middle = z2_middle.flatten()

# Calculate the Cartesian positions for the fourth link
x4_middle = (L1 * np.cos(theta1) * np.cos(theta2) + 
             PIP * np.cos(theta1 + theta3) * np.cos(theta2) + 
             L4 * np.cos(theta1 + theta3 + theta4) * np.cos(theta2))
y4_middle = (L1 * np.sin(theta1) + 
             PIP * np.sin(theta1 + theta3) + 
             L4 * np.sin(theta1 + theta3 + theta4))  # Offset by 10 units in the y direction
z4_middle = (L1 * np.cos(theta1) * np.sin(theta2) + 
             PIP * np.cos(theta1 + theta3) * np.sin(theta2) + 
             L4 * np.cos(theta1 + theta3 + theta4) * np.sin(theta2) + 2)

# Flatten the arrays for plotting
x4_middle = x4_middle.flatten()
y4_middle = y4_middle.flatten()
z4_middle = z4_middle.flatten()

# Calculate the Cartesian positions for the pinky finger's first link (Flexion/Extension)
x1_pinky = L1 * np.cos(theta1)
y1_pinky = L1 * np.sin(theta1)   # Offset by 20 units in the y direction
z1_pinky = np.random.uniform(-1e-3, 1e-3, x1_pinky.shape) + 4  # Small random perturbation

# Flatten the arrays for plotting
x1_pinky = x1_pinky.flatten()
y1_pinky = y1_pinky.flatten()
z1_pinky = z1_pinky.flatten()

# Calculate the Cartesian positions for the PIP link (Abduction/Adduction and Flexion/Extension)
x2_pinky = L1 * np.cos(theta1) * np.cos(theta2) + PIP * np.cos(theta1 + theta3) * np.cos(theta2)
y2_pinky = L1 * np.sin(theta1) + PIP * np.sin(theta1 + theta3)  # Offset by 20 units in the y direction
z2_pinky = L1 * np.cos(theta1) * np.sin(theta2) + PIP * np.cos(theta1 + theta3) * np.sin(theta2) + 4

# Flatten the arrays for plotting
x2_pinky = x2_pinky.flatten()
y2_pinky = y2_pinky.flatten()
z2_pinky = z2_pinky.flatten()

# Calculate the Cartesian positions for the fourth link
x4_pinky = (L1 * np.cos(theta1) * np.cos(theta2) + 
            PIP * np.cos(theta1 + theta3) * np.cos(theta2) + 
            L4 * np.cos(theta1 + theta3 + theta4) * np.cos(theta2))
y4_pinky = (L1 * np.sin(theta1) + 
            PIP * np.sin(theta1 + theta3) + 
            L4 * np.sin(theta1 + theta3 + theta4))  # Offset by 20 units in the y direction
z4_pinky = (L1 * np.cos(theta1) * np.sin(theta2) + 
            PIP * np.cos(theta1 + theta3) * np.sin(theta2) + 
            L4 * np.cos(theta1 + theta3 + theta4) * np.sin(theta2) + 4)

# Flatten the arrays for plotting
x4_pinky = x4_pinky.flatten()
y4_pinky = y4_pinky.flatten()
z4_pinky = z4_pinky.flatten()

# Calculate the Cartesian positions for the thumb's first link (MCP: Flexion/Extension and Rotation)
x1_thumb = L1_thumb * np.cos(theta1_thumb) * np.cos(theta2_thumb)-3.2
y1_thumb = L1_thumb * np.sin(theta1_thumb)
z1_thumb = L1_thumb * np.cos(theta1_thumb) * np.sin(theta2_thumb)+10

# Flatten the arrays for plotting
x1_thumb = x1_thumb.flatten()
y1_thumb = y1_thumb.flatten()
z1_thumb = z1_thumb.flatten()

# Calculate the Cartesian positions for the PIP link (Flexion/Extension)
x2_thumb = L1_thumb * np.cos(theta1_thumb) * np.cos(theta2_thumb) + L2_thumb * np.cos(theta1_thumb + theta3_thumb) * np.cos(theta2_thumb)-3.2
y2_thumb = L1_thumb * np.sin(theta1_thumb) + L2_thumb * np.sin(theta1_thumb + theta3_thumb)
z2_thumb = L1_thumb * np.cos(theta1_thumb) * np.sin(theta2_thumb) + L2_thumb * np.cos(theta1_thumb + theta3_thumb) * np.sin(theta2_thumb) + 10

# Flatten the arrays for plotting
x2_thumb = x2_thumb.flatten()
y2_thumb = y2_thumb.flatten()
z2_thumb = z2_thumb.flatten()

# Calculate the Cartesian positions for the DIP link (Flexion/Extension)
x3_thumb = (L1_thumb * np.cos(theta1_thumb) * np.cos(theta2_thumb) + 
            L2_thumb * np.cos(theta1_thumb + theta3_thumb) * np.cos(theta2_thumb) + 
            L3_thumb * np.cos(theta1_thumb + theta3_thumb + theta4_thumb) * np.cos(theta2_thumb)-3.2)
y3_thumb = (L1_thumb * np.sin(theta1_thumb) + 
            L2_thumb * np.sin(theta1_thumb + theta3_thumb) + 
            L3_thumb * np.sin(theta1_thumb + theta3_thumb + theta4_thumb))
z3_thumb = (L1_thumb * np.cos(theta1_thumb) * np.sin(theta2_thumb) + 
            L2_thumb * np.cos(theta1_thumb + theta3_thumb) * np.sin(theta2_thumb) + 
            L3_thumb * np.cos(theta1_thumb + theta3_thumb + theta4_thumb) * np.sin(theta2_thumb) + 10)

# Flatten the arrays for plotting
x3_thumb = x3_thumb.flatten()
y3_thumb = y3_thumb.flatten()
z3_thumb = z3_thumb.flatten()

# Center of thumb's reachability space before rotation
center_x_thumb = np.mean(x1_thumb)
center_y_thumb = np.mean(y1_thumb)
center_z_thumb = np.mean(z1_thumb)

# Translate thumb points to origin
x1_thumb -= center_x_thumb
y1_thumb -= center_y_thumb
z1_thumb -= center_z_thumb
x2_thumb -= center_x_thumb
y2_thumb -= center_y_thumb
z2_thumb -= center_z_thumb
x3_thumb -= center_x_thumb
y3_thumb -= center_y_thumb
z3_thumb -= center_z_thumb

# Rotation matrix for rotating about the y-axis by 90 degrees
rotation_matrix = np.array([
    [np.cos(-np.pi / 2), 0, np.sin(-np.pi / 2)],
    [0, 1, 0],
    [-np.sin(-np.pi / 2), 0, np.cos(-np.pi / 2)]
])

# Apply the rotation to the thumb positions
points1_thumb_rotated = np.dot(rotation_matrix, np.vstack((x1_thumb, y1_thumb, z1_thumb)))
points2_thumb_rotated = np.dot(rotation_matrix, np.vstack((x2_thumb, y2_thumb, z2_thumb)))
points3_thumb_rotated = np.dot(rotation_matrix, np.vstack((x3_thumb, y3_thumb, z3_thumb)))

# Translate thumb points back to their original position
x1_thumb = points1_thumb_rotated[0, :] + center_x_thumb
y1_thumb = points1_thumb_rotated[1, :] + center_y_thumb
z1_thumb = points1_thumb_rotated[2, :] + center_z_thumb
x2_thumb = points2_thumb_rotated[0, :] + center_x_thumb
y2_thumb = points2_thumb_rotated[1, :] + center_y_thumb
z2_thumb = points2_thumb_rotated[2, :] + center_z_thumb
x3_thumb = points3_thumb_rotated[0, :] + center_x_thumb
y3_thumb = points3_thumb_rotated[1, :] + center_y_thumb
z3_thumb = points3_thumb_rotated[2, :] + center_z_thumb

# Create point clouds for all links
points1_index = np.vstack((x1_index, y1_index, z1_index)).T
points2_index = np.vstack((x2_index, y2_index, z2_index)).T
points4_index = np.vstack((x4_index, y4_index, z4_index)).T

points1_middle = np.vstack((x1_middle, y1_middle, z1_middle)).T
points2_middle = np.vstack((x2_middle, y2_middle, z2_middle)).T
points4_middle = np.vstack((x4_middle, y4_middle, z4_middle)).T

points1_pinky = np.vstack((x1_pinky, y1_pinky, z1_pinky)).T
points2_pinky = np.vstack((x2_pinky, y2_pinky, z2_pinky)).T
points4_pinky = np.vstack((x4_pinky, y4_pinky, z4_pinky)).T

points1_thumb = np.vstack((x1_thumb, y1_thumb, z1_thumb)).T
points2_thumb = np.vstack((x2_thumb, y2_thumb, z2_thumb)).T
points3_thumb = np.vstack((x3_thumb, y3_thumb, z3_thumb)).T

# Compute the convex hulls for creating surface meshes
hull1_index = ConvexHull(points1_index)
hull2_index = ConvexHull(points2_index)
hull4_index = ConvexHull(points4_index)

hull1_middle = ConvexHull(points1_middle)
hull2_middle = ConvexHull(points2_middle)
hull4_middle = ConvexHull(points4_middle)

hull1_pinky = ConvexHull(points1_pinky)
hull2_pinky = ConvexHull(points2_pinky)
hull4_pinky = ConvexHull(points4_pinky)

hull1_thumb = ConvexHull(points1_thumb)
hull2_thumb = ConvexHull(points2_thumb)
hull3_thumb = ConvexHull(points3_thumb)

# Get the vertices of the convex hulls
hull_points1_index = points1_index[hull1_index.vertices]
hull_points2_index = points2_index[hull2_index.vertices]
hull_points4_index = points4_index[hull4_index.vertices]

hull_points1_middle = points1_middle[hull1_middle.vertices]
hull_points2_middle = points2_middle[hull2_middle.vertices]
hull_points4_middle = points4_middle[hull4_middle.vertices]

hull_points1_pinky = points1_pinky[hull1_pinky.vertices]
hull_points2_pinky = points2_pinky[hull2_pinky.vertices]
hull_points4_pinky = points4_pinky[hull4_pinky.vertices]

hull_points1_thumb = points1_thumb[hull1_thumb.vertices]
hull_points2_thumb = points2_thumb[hull2_thumb.vertices]
hull_points3_thumb = points3_thumb[hull3_thumb.vertices]

# Calculate the opposition space for pairs of points
def calculate_opposition_space(hull1, hull2):
    combined_points = np.concatenate((hull1, hull2))
    hull_combined = ConvexHull(combined_points)
    return hull_combined.volume

# Calculate the opposition spaces for each pair of segments
opposition_spaces = np.zeros((15, 15))

hull_points_list = [hull_points1_index, hull_points2_index, hull_points4_index,
                    hull_points1_middle, hull_points2_middle, hull_points4_middle,
                    hull_points1_pinky, hull_points2_pinky, hull_points4_pinky,
                    hull_points1_thumb, hull_points2_thumb, hull_points3_thumb]

# Calculate opposition spaces for each pair of segments
for i in range(len(hull_points_list)):
    for j in range(i + 1, len(hull_points_list)):
        opposition_spaces[i, j] = calculate_opposition_space(hull_points_list[i], hull_points_list[j])

# Create a heatmap to visualize the opposition spaces
plt.figure(figsize=(10, 8))
sns.heatmap(opposition_spaces, annot=True, fmt=".0f", cmap="Blues", xticklabels=False, yticklabels=False)
plt.title("Opposition Spaces (d = 30mm)")
plt.xlabel("Hand Link")
plt.ylabel("Hand Link")
plt.show()
