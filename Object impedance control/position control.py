import mujoco_py
import numpy as np
import os

# Load the MuJoCo model from the XML string
model_path = "/home/linux/Documents/mujoco200_linux/grasp/manipulator45.xml" # Replace with the actual path to your XML file
model = mujoco_py.load_model_from_path(model_path)
sim = mujoco_py.MjSim(model)

# Viewer for visualization
viewer = mujoco_py.MjViewer(sim)

# Set target positions for the joints
target_positions = {
    "1": -1.047,  # Target position for joint 1 (in radians)
    "2": -1.57,  # Target position for joint 2 (in radians)
    "3": 1.047,  # Target position for joint 3 (in radians)
    "4": 1.57   # Target position for joint 4 (in radians)
}

# Set the position control gains (kp)
kp = 100.0

# Simulation loop
while True:
    # Get the current joint positions
    joint_positions = {name: sim.data.get_joint_qpos(name) for name in target_positions}

    # Lists to hold current positions, target positions, and errors
    current_positions = []
    desired_positions = []
    errors = []

    # Calculate the control signals and populate the lists
    controls = []
    for name, target_pos in target_positions.items():
        current_pos = joint_positions[name]
        error = target_pos - current_pos
        control = kp * error
        controls.append(control)

        # Append values to the lists
        current_positions.append(current_pos)
        desired_positions.append(target_pos)
        errors.append(error)

    # Print the lists as arrays
    print(f"Current Positions (rad): {np.array(current_positions)}")
    print(f"Target Positions (rad): {np.array(desired_positions)}")
    print(f"Errors (rad): {np.array(errors)}")
    print()  # Add an empty line for better readability

    # Apply the control signals
    sim.data.ctrl[:] = controls

    # Step the simulation forward
    sim.step()

    # Render the scene
    viewer.render()
