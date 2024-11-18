import numpy as np
import pyvista as pv

def create_finger(base, joint_angles, segment_lengths):
    """
    Create a four-joint finger based on joint angles and segment lengths.
    Args:
        base: The starting point of the finger (numpy array).
        joint_angles: Array of joint angles [theta1, theta2, theta3].
        segment_lengths: Array of segment lengths [l1, l2, l3, l4].
    Returns:
        numpy array of points representing the finger.
    """
    points = [base]
    current_position = base
    current_angle = 0
    
    for i in range(len(joint_angles)):
        current_angle += joint_angles[i]
        dx = segment_lengths[i] * np.cos(current_angle)
        dy = segment_lengths[i] * np.sin(current_angle)
        next_point = current_position + np.array([dx, dy, 0])
        points.append(next_point)
        current_position = next_point
    
    # Add the fingertip
    fingertip = current_position + np.array([segment_lengths[-1], 0, 0])
    points.append(fingertip)
    return np.array(points)

# Step 1: Define Finger Parameters
# Base positions for the index and thumb
index_base = np.array([0, 0, 0])
thumb_base = np.array([0, 0, 0])

# Joint angles for the pinch grasp
index_angles = [45, 30, 30, -30]  # Flex the index finger
thumb_angles = [-45, -30, -30, 30]    # Flex the thumb

# Segment lengths
finger_lengths = [0.02, 0.01, 0.008, 0.005]

# Create index and thumb fingers
index_finger_points = create_finger(index_base, index_angles, finger_lengths)
thumb_finger_points = create_finger(thumb_base, thumb_angles, finger_lengths)

# Step 2: Visualize the Fingers
plotter = pv.Plotter()

# Add fingers
index_finger_mesh = pv.lines_from_points(index_finger_points)
thumb_finger_mesh = pv.lines_from_points(thumb_finger_points)

plotter.add_mesh(index_finger_mesh, color='blue', line_width=4, label='Index Finger')
plotter.add_mesh(thumb_finger_mesh, color='red', line_width=4, label='Thumb Finger')

# Step 3: Add a Grasped Object
object_center = (index_finger_points[-1] + thumb_finger_points[-1]) / 2
object_radius = 0.005
grasped_object = pv.Sphere(center=object_center, radius=object_radius)

plotter.add_mesh(grasped_object, color='green', label='Grasped Object')

# Step 4: Show Visualization
plotter.add_legend()
plotter.show()
