import numpy as np
import pyvista as pv
from scipy.spatial import ConvexHull

# Finger lengths and joint angle limits
L1, PIP, L4 = 0.035, 0.045, 0.063  # Link lengths for the fingers
L1_thumb, L2_thumb, L3_thumb = 0.05, 0.035, 0.05  # Link lengths for the thumb

# Define the pinch grasp position for index and thumb fingers
# Assuming specific joint angles for pinch grasp
theta1_index_grasp, theta2_index_grasp, theta3_index_grasp, theta4_index_grasp = np.deg2rad([45, 0, 70, 30])
theta1_thumb_grasp, theta2_thumb_grasp, theta3_thumb_grasp, theta4_thumb_grasp = np.deg2rad([45, 0, 60, 30])

# Calculate the pinch grasp fingertip positions
x_index_grasp = (L1 * np.cos(theta1_index_grasp) * np.cos(theta2_index_grasp) +
                 PIP * np.cos(theta1_index_grasp + theta3_index_grasp) * np.cos(theta2_index_grasp) +
                 L4 * np.cos(theta1_index_grasp + theta3_index_grasp + theta4_index_grasp) * np.cos(theta2_index_grasp))
y_index_grasp = (L1 * np.sin(theta1_index_grasp) +
                 PIP * np.sin(theta1_index_grasp + theta3_index_grasp) +
                 L4 * np.sin(theta1_index_grasp + theta3_index_grasp + theta4_index_grasp))
z_index_grasp = (L1 * np.cos(theta1_index_grasp) * np.sin(theta2_index_grasp) +
                 PIP * np.cos(theta1_index_grasp + theta3_index_grasp) * np.sin(theta2_index_grasp) +
                 L4 * np.cos(theta1_index_grasp + theta3_index_grasp + theta4_index_grasp) * np.sin(theta2_index_grasp))

x_thumb_grasp = (L1_thumb * np.cos(theta1_thumb_grasp) * np.cos(theta2_thumb_grasp) +
                 L2_thumb * np.cos(theta1_thumb_grasp + theta3_thumb_grasp) * np.cos(theta2_thumb_grasp) +
                 L3_thumb * np.cos(theta1_thumb_grasp + theta3_thumb_grasp + theta4_thumb_grasp) * np.cos(theta2_thumb_grasp))
y_thumb_grasp = (L1_thumb * np.sin(theta1_thumb_grasp) +
                 L2_thumb * np.sin(theta1_thumb_grasp + theta3_thumb_grasp) +
                 L3_thumb * np.sin(theta1_thumb_grasp + theta3_thumb_grasp + theta4_thumb_grasp))
z_thumb_grasp = (L1_thumb * np.cos(theta1_thumb_grasp) * np.sin(theta2_thumb_grasp) +
                 L2_thumb * np.cos(theta1_thumb_grasp + theta3_thumb_grasp) * np.sin(theta2_thumb_grasp) +
                 L3_thumb * np.cos(theta1_thumb_grasp + theta3_thumb_grasp + theta4_thumb_grasp) * np.sin(theta2_thumb_grasp))

# Define the region occupied by the pinch grasp
occupied_radius = 1.5  # Define a suitable radius
pinch_grasp_region = np.array([x_index_grasp, y_index_grasp, z_index_grasp, x_thumb_grasp, y_thumb_grasp, z_thumb_grasp])

# Filter workspace function to avoid pinch grasp region
def filter_workspace(x, y, z, pinch_grasp_region, radius):
    mask = np.sqrt((x - pinch_grasp_region[0])**2 +
                   (y - pinch_grasp_region[1])**2 +
                   (z - pinch_grasp_region[2])**2) > radius
    return x[mask], y[mask], z[mask]

# Generate reachability map for middle and pinky fingers
def generate_reachability_map(theta1_range, theta2_range, theta3_range, L1, L2, L3):
    x_reach, y_reach, z_reach = [], [], []
    for theta1 in theta1_range:
        for theta2 in theta2_range:
            for theta3 in theta3_range:
                x = (L1 * np.cos(theta1) * np.cos(theta2) +
                     L2 * np.cos(theta1 + theta3) * np.cos(theta2) +
                     L3 * np.cos(theta1 + theta3) * np.cos(theta2))
                y = (L1 * np.sin(theta1) +
                     L2 * np.sin(theta1 + theta3) +
                     L3 * np.sin(theta1 + theta3))
                z = (L1 * np.cos(theta1) * np.sin(theta2) +
                     L2 * np.cos(theta1 + theta3) * np.sin(theta2) +
                     L3 * np.cos(theta1 + theta3) * np.sin(theta2))
                x_reach.append(x)
                y_reach.append(y)
                z_reach.append(z)
    return np.array(x_reach), np.array(y_reach), np.array(z_reach)

# Define joint ranges
theta1_range = np.linspace(-np.pi/2, np.pi/2, 20)
theta2_range = np.linspace(-np.pi/4, np.pi/4, 20)
theta3_range = np.linspace(-np.pi/6, np.pi/6, 20)

# Calculate reachability map for middle and pinky fingers
x_middle, y_middle, z_middle = generate_reachability_map(theta1_range, theta2_range, theta3_range, L1, PIP, L4)
x_pinky, y_pinky, z_pinky = generate_reachability_map(theta1_range, theta2_range, theta3_range, L1, PIP, L4)

# Filter out the pinch grasp region
x_middle_filtered, y_middle_filtered, z_middle_filtered = filter_workspace(x_middle, y_middle, z_middle, pinch_grasp_region, occupied_radius)
x_pinky_filtered, y_pinky_filtered, z_pinky_filtered = filter_workspace(x_pinky, y_pinky, z_pinky, pinch_grasp_region, occupied_radius)

# Visualize the remaining workspace using PyVista
points_middle = np.vstack((x_middle_filtered, y_middle_filtered, z_middle_filtered)).T
points_pinky = np.vstack((x_pinky_filtered, y_pinky_filtered, z_pinky_filtered)).T

cloud_middle = pv.PolyData(points_middle)
cloud_pinky = pv.PolyData(points_pinky)

plotter = pv.Plotter()
plotter.add_mesh(cloud_middle, color='blue', point_size=1, render_points_as_spheres=True, label='Middle Finger')
plotter.add_mesh(cloud_pinky, color='green', point_size=1, render_points_as_spheres=True, label='Pinky Finger')

plotter.add_legend()
plotter.show()
