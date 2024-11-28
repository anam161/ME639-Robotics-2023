import mujoco_py
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Load the Mujoco model
model = mujoco_py.load_model_from_path('/home/linux/Documents/mujoco200_linux/leap hand ws/leapws (copy).xml')  # Replace with your XML file path
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)

# Define joint limits for MCP, PIP, DIP, and Fingertip (use appropriate ranges from your model)
joint_ranges = {
    "mcp_joint1_motor": (-0.04, 0),  # Example MCP joint range
    "pip_motor": (0, 0),                # Example PIP joint range
    "dip_motor": (-0.02, 0),         # Example DIP joint range
    "fingertip_motor": (-0.02, 0),    # Example Fingertip joint range
    "pip_4_motor": (0.04, 0),
    "thumb_pip_4_motor": (0, 0),  # Example Thumb PIP joint range
    "thumb_dip_4_motor": (-0.02, 0),  # Example Thumb DIP joint range
    "thumb_fingertip_4_motor": (-0.02, 0)
}

# Create a mesh grid of joint angles
mcp_angles = np.linspace(joint_ranges["mcp_joint1_motor"][0], joint_ranges["mcp_joint1_motor"][1], 3)
pip_angles = np.linspace(joint_ranges["pip_motor"][0], joint_ranges["pip_motor"][1], 3)
dip_angles = np.linspace(joint_ranges["dip_motor"][0], joint_ranges["dip_motor"][1], 3)
fingertip_angles = np.linspace(joint_ranges["fingertip_motor"][0], joint_ranges["fingertip_motor"][1], 3)

thumb_pip1_angles = np.linspace(joint_ranges["pip_4_motor"][0], joint_ranges["pip_4_motor"][1], 3)
thumb_pip2_angles = np.linspace(joint_ranges["thumb_pip_4_motor"][0], joint_ranges["thumb_pip_4_motor"][1], 3)
thumb_dip_angles = np.linspace(joint_ranges["thumb_dip_4_motor"][0], joint_ranges["thumb_dip_4_motor"][1], 3)
thumb_fingertip_angles = np.linspace(joint_ranges["thumb_fingertip_4_motor"][0], joint_ranges["thumb_fingertip_4_motor"][1], 3)


# Store fingertip positions in the workspace
index_workspace = []
thumb_workspace = []

# Simulate and record fingertip positions
for mcp in mcp_angles:
    for pip in pip_angles:
        for dip in dip_angles:
            for fingertip in fingertip_angles:
                for thumb_pip1 in thumb_pip1_angles:
                    for thumb_pip2 in thumb_pip2_angles:
                        for thumb_dip in thumb_dip_angles:
                            for thumb_fingertip in thumb_fingertip_angles:
                                # Set joint positions
                                sim.data.ctrl[0] = mcp  # MCP joint motor
                                sim.data.ctrl[1] = pip  # PIP joint motor
                                sim.data.ctrl[2] = dip  # DIP joint motor
                                sim.data.ctrl[3] = fingertip  # Fingertip motor

                                 # Set joint positions for thumb
                                sim.data.ctrl[4] = thumb_pip1
                                sim.data.ctrl[5] = thumb_pip2  # Thumb PIP motor
                                sim.data.ctrl[6] = thumb_dip  # Thumb DIP motor
                                sim.data.ctrl[7] = thumb_fingertip  # Thumb Fingertip motor


                                # Step the simulation
                                sim.step()

                                # Get fingertip position
                                fingertip_pos1 = sim.data.site_xpos[sim.model.site_name2id('tip1')].copy()
                                fingertip_pos2 = sim.data.site_xpos[sim.model.site_name2id('tip4')].copy()
                                # Store the position in workspace list
                                index_workspace.append(fingertip_pos1)
                                thumb_workspace.append(fingertip_pos2)

                                # Update viewer to visualize the movement
                                viewer.render()

# Convert the workspace to a numpy array for easy handling
index_workspace = np.array(index_workspace)
thumb_workspace = np.array(thumb_workspace)

np.save("Workspace", index_workspace)
np.save("Thumb_Workspace", thumb_workspace)

# Calculate the Convex Hull of the combined workspace and its volume
combined_workspace = np.vstack((index_workspace, thumb_workspace))
try:
    hull = ConvexHull(combined_workspace)
    workspace_volume = hull.volume
    print(f"Combined Workspace Volume: {workspace_volume} cubic units")
except Exception as e:
    print("Error computing Convex Hull:", e)
    workspace_volume = None

# Get object geometry
object_geom_id = sim.model.geom_name2id('box_geom')  # Replace 'box' with your object name
object_size = sim.model.geom_size[object_geom_id]  # Half extents of the box
object_position = sim.model.geom_pos[object_geom_id]  # Position of the object

# Calculate the corner points of the box (8 corners)
offsets = np.array([
    [-1, -1, -1],
    [-1, -1,  1],
    [-1,  1, -1],
    [-1,  1,  1],
    [ 1, -1, -1],
    [ 1, -1,  1],
    [ 1,  1, -1],
    [ 1,  1,  1]
]) * object_size  # Scale by the half extents
box_corners = offsets + object_position  # Translate to object position

# Create the convex hull for the object
object_hull = ConvexHull(box_corners)
object_volume = object_hull.volume
print(f"Object Convex Hull Volume: {object_volume} cubic units")

# Sample points inside workspace convex hull
workspace_delaunay = Delaunay(combined_workspace)
object_delaunay = Delaunay(box_corners)

# Generate a dense grid within workspace bounds
min_bounds = np.min(combined_workspace, axis=0)
max_bounds = np.max(combined_workspace, axis=0)
grid_x, grid_y, grid_z = np.mgrid[
    min_bounds[0]:max_bounds[0]:50j,
    min_bounds[1]:max_bounds[1]:50j,
    min_bounds[2]:max_bounds[2]:50j
]
grid_points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

# Find points inside both hulls (intersection points)
inside_workspace = workspace_delaunay.find_simplex(grid_points) >= 0
inside_object = object_delaunay.find_simplex(grid_points) >= 0
intersection_points = grid_points[inside_workspace & inside_object]

# Compute intersection convex hull and its volume
if len(intersection_points) > 3:  # Minimum points to form a convex hull
    intersection_hull = ConvexHull(intersection_points)
    intersection_volume = intersection_hull.volume
else:
    intersection_volume = 0
print(f"Intersection Volume: {intersection_volume} cubic units")

# Calculate remaining workspace volume
remaining_workspace_volume = workspace_volume - intersection_volume
print(f"Remaining Workspace Volume: {remaining_workspace_volume} cubic units")

# Plot the workspace, object, and intersection
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot index finger workspace
ax.scatter(index_workspace[:, 0], index_workspace[:, 1], index_workspace[:, 2], c='r', marker='o', label='Index Finger')
# Plot thumb workspace
ax.scatter(thumb_workspace[:, 0], thumb_workspace[:, 1], thumb_workspace[:, 2], c='b', marker='^', label='Thumb')

for simplex in hull.simplices:
    triangle = combined_workspace[simplex]
    poly3d = [[tuple(triangle[0]), tuple(triangle[1]), tuple(triangle[2])]]
    ax.add_collection3d(Poly3DCollection(poly3d, color='cyan', alpha=0.6))

# Plot object convex hull
ax.scatter(box_corners[:, 0], box_corners[:, 1], box_corners[:, 2], c='b', marker='^', label="Object Points")
for simplex in object_hull.simplices:
    triangle = box_corners[simplex]
    poly3d = [[tuple(triangle[0]), tuple(triangle[1]), tuple(triangle[2])]]
    ax.add_collection3d(Poly3DCollection(poly3d, color='green', alpha=0.5))

# Plot intersection points
if len(intersection_points) > 0:
    ax.scatter(intersection_points[:, 0], intersection_points[:, 1], intersection_points[:, 2], c='purple', label="Intersection Points")

# Set axis limits for better visualization
ax.set_xlim(-0.2, 0.2)
ax.set_ylim(-0.2, 0.2)
ax.set_zlim(0, 0.2)

# Add labels and legend
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
plt.title('Workspace, Object, and Intersection Volumes')
plt.legend()
plt.show()
