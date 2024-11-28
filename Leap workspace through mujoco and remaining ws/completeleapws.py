import mujoco_py
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Load the Mujoco model
model = mujoco_py.load_model_from_path('/home/linux/Documents/mujoco200_linux/leap hand ws/leapws (copy)1.xml')
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)

# Define joint limits for MCP, PIP, DIP, Fingertip, and Thumb (use appropriate ranges from your model)
joint_ranges = {
    "mcp_joint1_motor": (-0.08, 0),  # Example MCP joint range
    "pip_motor": (0, 0),          # Example PIP joint range
    "dip_motor": (-0.015, 0),        # Example DIP joint range
    "fingertip_motor": (-0.015, 0),  # Example Fingertip joint range
    "mcp_joint_2_motor": (-0.08, 0),  # Example MCP joint range
    "pip_2_motor": (0, 0),          # Example PIP joint range
    "dip_2_motor": (-0.015, 0),        # Example DIP joint range
    "fingertip_2_motor": (-0.015, 0), 
    "mcp_joint_3_motor": (-0.08, 0),  # Example MCP joint range
    "pip_3_motor": (0, 0),          # Example PIP joint range
    "dip_3_motor": (-0.015, 0),        # Example DIP joint range
    "fingertip_3_motor": (-0.015, 0),
    "pip_4_motor": (0.09, 0),
    "thumb_pip_4_motor": (0, 0),  # Example Thumb PIP joint range
    "thumb_dip_4_motor": (-0.08, 0),  # Example Thumb DIP joint range
    "thumb_fingertip_4_motor": (-0.08, 0)  # Example Thumb Fingertip joint range
}

# Create a mesh grid of joint angles for both fingers
mcp_angles = np.linspace(joint_ranges["mcp_joint1_motor"][0], joint_ranges["mcp_joint1_motor"][1], 1)
pip_angles = np.linspace(joint_ranges["pip_motor"][0], joint_ranges["pip_motor"][1], 1)
dip_angles = np.linspace(joint_ranges["dip_motor"][0], joint_ranges["dip_motor"][1], 2)
fingertip_angles = np.linspace(joint_ranges["fingertip_motor"][0], joint_ranges["fingertip_motor"][1], 2)

mcp2_angles = np.linspace(joint_ranges["mcp_joint_2_motor"][0], joint_ranges["mcp_joint_2_motor"][1], 2)
pip2_angles = np.linspace(joint_ranges["pip_2_motor"][0], joint_ranges["pip_2_motor"][1], 1)
dip2_angles = np.linspace(joint_ranges["dip_2_motor"][0], joint_ranges["dip_2_motor"][1], 1)
fingertip2_angles = np.linspace(joint_ranges["fingertip_2_motor"][0], joint_ranges["fingertip_2_motor"][1], 2)

mcp3_angles = np.linspace(joint_ranges["mcp_joint_3_motor"][0], joint_ranges["mcp_joint_3_motor"][1], 2)
pip3_angles = np.linspace(joint_ranges["pip_3_motor"][0], joint_ranges["pip_3_motor"][1], 2)
dip3_angles = np.linspace(joint_ranges["dip_3_motor"][0], joint_ranges["dip_3_motor"][1], 2)
fingertip3_angles = np.linspace(joint_ranges["fingertip_3_motor"][0], joint_ranges["fingertip_3_motor"][1], 2)

thumb_pip1_angles = np.linspace(joint_ranges["pip_4_motor"][0], joint_ranges["pip_4_motor"][1], 2)
thumb_pip2_angles = np.linspace(joint_ranges["thumb_pip_4_motor"][0], joint_ranges["thumb_pip_4_motor"][1], 2)
thumb_dip_angles = np.linspace(joint_ranges["thumb_dip_4_motor"][0], joint_ranges["thumb_dip_4_motor"][1], 2)
thumb_fingertip_angles = np.linspace(joint_ranges["thumb_fingertip_4_motor"][0], joint_ranges["thumb_fingertip_4_motor"][1], 2)

# Store fingertip and thumb positions in the workspace
index_workspace = []
middle_workspace=[]
pinky_workspace=[]
thumb_workspace = []

# Simulate and record fingertip and thumb positions
for mcp in mcp_angles:
    for pip in pip_angles:
        for dip in dip_angles:
            for fingertip in fingertip_angles:
                for mcp2 in mcp2_angles:
                    for pip2 in pip2_angles:
                        for dip2 in dip2_angles:
                            for fingertip2 in fingertip2_angles:
                                for mcp3 in mcp3_angles:
                                    for pip3 in pip3_angles:
                                        for dip3 in dip3_angles:
                                            for fingertip3 in fingertip3_angles:
                                                for thumb_pip1 in thumb_pip1_angles:
                                                    for thumb_pip2 in thumb_pip2_angles:
                                                        for thumb_dip in thumb_dip_angles:
                                                            for thumb_fingertip in thumb_fingertip_angles:
                                                                # Set joint positions for index finger
                                                                sim.data.ctrl[0] = mcp  # MCP joint motor
                                                                sim.data.ctrl[1] = pip  # PIP joint motor
                                                                sim.data.ctrl[2] = dip  # DIP joint motor
                                                                sim.data.ctrl[3] = fingertip  # Fingertip motor
                                                                # Set joint positions for index finger
                                                                sim.data.ctrl[4] = mcp2  # MCP joint motor
                                                                sim.data.ctrl[5] = pip2  # PIP joint motor
                                                                sim.data.ctrl[6] = dip2  # DIP joint motor
                                                                sim.data.ctrl[7] = fingertip2  # Fingertip motor

                                                                 # Set joint positions for index finger
                                                                sim.data.ctrl[8] = mcp3  # MCP joint motor
                                                                sim.data.ctrl[9] = pip3  # PIP joint motor
                                                                sim.data.ctrl[10] = dip3  # DIP joint motor
                                                                sim.data.ctrl[11] = fingertip3  # Fingertip motor

                                                                # Set joint positions for thumb
                                                                sim.data.ctrl[12] = thumb_pip1
                                                                sim.data.ctrl[13] = thumb_pip2  # Thumb PIP motor
                                                                sim.data.ctrl[14] = thumb_dip  # Thumb DIP motor
                                                                sim.data.ctrl[15] = thumb_fingertip  # Thumb Fingertip motor

                                                                # Step the simulation
                                                                sim.step()

                                                                # Get fingertip positions
                                                                fingertip1_pos = sim.data.site_xpos[sim.model.site_name2id('tip1')].copy()
                                                                fingertip2_pos = sim.data.site_xpos[sim.model.site_name2id('tip2')].copy()
                                                                fingertip3_pos = sim.data.site_xpos[sim.model.site_name2id('tip3')].copy()
                                                                thumb_pos = sim.data.site_xpos[sim.model.site_name2id('tip4')].copy()

                                                                # Store the positions in respective workspace lists
                                                                index_workspace.append(fingertip1_pos)
                                                                middle_workspace.append(fingertip2_pos)
                                                                pinky_workspace.append(fingertip3_pos)
                                                                thumb_workspace.append(thumb_pos)

                                                                # Update viewer to visualize the movement
                                                                viewer.render()

# Convert the workspace to numpy arrays for easy handling
index_workspace = np.array(index_workspace)
middle_workspace = np.array(middle_workspace)
pinky_workspace = np.array(pinky_workspace)
thumb_workspace = np.array(thumb_workspace)

np.save("Workspace", index_workspace)
np.save("Workspace", middle_workspace)
np.save("Workspace", pinky_workspace)
np.save("Thumb_Workspace", thumb_workspace)

print("Workspace shape:", index_workspace.shape)
print("Workspace shape:", middle_workspace.shape)
print("Workspace shape:", pinky_workspace.shape)
print("Thumb Workspace shape:", thumb_workspace.shape)

# Calculate the Convex Hull of the combined workspace and its volume
combined_workspace = np.vstack((index_workspace,middle_workspace,pinky_workspace, thumb_workspace))
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

# Plot the workspace (3D scatter plot)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot index finger workspace
ax.scatter(index_workspace[:, 0], index_workspace[:, 1], index_workspace[:, 2], c='r', marker='o', label='Index Finger')
# Plot index finger workspace
ax.scatter(middle_workspace[:, 0], middle_workspace[:, 1], middle_workspace[:, 2], c='r', marker='o', label='middle Finger')

# Plot index finger workspace
ax.scatter(pinky_workspace[:, 0], pinky_workspace[:, 1], pinky_workspace[:, 2], c='r', marker='o', label='pinky Finger')
# Plot thumb workspace
ax.scatter(thumb_workspace[:, 0], thumb_workspace[:, 1], thumb_workspace[:, 2], c='b', marker='^', label='Thumb')

# Add Convex Hull faces to visualize the workspace volume
for simplex in hull.simplices:
    # Use workspace points forming each facet
    triangle = combined_workspace[simplex]
    poly3d = [[tuple(triangle[0]), tuple(triangle[1]), tuple(triangle[2])]]
    ax.add_collection3d(Poly3DCollection(poly3d, color='cyan', alpha=0.5))  # Corrected usage of Poly3DCollection

# Plot object convex hull
ax.scatter(box_corners[:, 0], box_corners[:, 1], box_corners[:, 2], c='b', marker='^', label="Object Points")
for simplex in object_hull.simplices:
    triangle = box_corners[simplex]
    poly3d = [[tuple(triangle[0]), tuple(triangle[1]), tuple(triangle[2])]]
    ax.add_collection3d(Poly3DCollection(poly3d, color='green', alpha=0.5))

# Plot intersection points
if len(intersection_points) > 0:
    ax.scatter(intersection_points[:, 0], intersection_points[:, 1], intersection_points[:, 2], c='purple', label="Intersection Points")


ax.set_xlim(-0.1, 0.15)
ax.set_ylim(-0.1, 0.15)
ax.set_zlim(-0.1, 0.15)

ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')

plt.title('Leap Hand Combined Workspace')
plt.legend()
plt.show()
