# -*- coding: utf-8 -*-
"""Untitled53.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1kBJLSgk43zilTbmuiSuqwk6eFCuDd7xK
"""

import numpy as np
import mujoco_py
import time

def grasp_matrix_2d(contact_points):
    G_list = []
    for t, n, x in contact_points:
        t = np.atleast_2d(t).T
        n = np.atleast_2d(n).T
        x = np.atleast_2d(x).T
        G_i = np.block([[t, n], [np.cross(x.T, t.T).T, np.cross(x.T, n.T).T]])
        G_list.append(G_i)
    G = np.hstack(G_list)
    return G

def W(t, n):
    t = np.atleast_2d(t).T
    n = np.atleast_2d(n).T
    w  = np.block([t, n])
    return w

def rotation_matrix_2d(theta):
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    return R

def jacobian(l1, l2, theta1, theta2):
    J = np.array([[-l1*np.sin(theta1), -l2*np.sin(theta1-theta2)],
                  [l1*np.cos(theta1), l2*np.cos(theta1-theta2)]])
    return J

def hand_jacobian(t, n, w, R, J):
    WRJ = np.dot(np.dot(w.T, R), J)
    return WRJ

# Contact points for grasp matrix
contact_points = [
    (np.array([0, -1]), np.array([1, 0]), np.array([-0.2, 0])),
    (np.array([0, 1]), np.array([-1, 0]), np.array([0.2, 0]))
]

G = grasp_matrix_2d(contact_points)

# Load the Mujoco model
model = mujoco_py.load_model_from_path("/home/linux/Documents/mujoco200_linux/grasp/manipulator45.xml")
sim = mujoco_py.MjSim(model)

# Define controller gains
Kp = 10*np.array([1, 1, 1])

# Desired positions and velocities
Xd = np.array([0, 0, 0])

# Initialize the simulation
sim_state = sim.get_state()
sim.set_state(sim_state)
sim.forward()

# Initialize the viewer
viewer = mujoco_py.MjViewer(sim)



# Simulation loop
for i in range(10000):
  # Get the current state
    X_full = sim.data.get_body_xpos('object')
    x, y = X_full[:2]
    theta = 0  # Replace with actual orientation extraction if necessary
    X = np.array([x, y, theta])

    # Initial angles
    initial_theta1_1 = 2.094
    initial_theta2_1 = 0.523599
    initial_theta1_2 = 2.094
    initial_theta2_2 = 0.523599

    # Apply external force to the object and update joint angles
    if 1500 <= i <= 1700:
        force = np.array([0, 0, 0, 0, 0, 0])

        # Update joint angles based on simulation state after force application
        theta1_1 = sim.data.qpos[model.joint_name2id('1')]  # Replace with actual joint name
        theta2_1 = sim.data.qpos[model.joint_name2id('2')]  # Replace with actual joint name
        theta1_2 = sim.data.qpos[model.joint_name2id('3')]  # Replace with actual joint name
        theta2_2 = sim.data.qpos[model.joint_name2id('4')]  # Replace with actual joint name

        # Print the joint angles during force application
        print(f"Step: {i}, Joint Angles: theta1_1 = {theta1_1}, theta2_1 = {theta2_1}, theta1_2 = {theta1_2}, theta2_2 = {theta2_2}")
    else:
        force = np.array([0, 0, 0, 0, 0, 0])
        sim.data.xfrc_applied[sim.model.body_name2id('object'), :] = force

        # Maintain initial joint angles by adding a fixed value to compensate for drift
        theta1_1 = initial_theta1_1
        theta2_1 = initial_theta2_1
        theta1_2 = initial_theta1_2
        theta2_2 = initial_theta2_2

        # # Correct for any deviations by setting qpos back to initial values
        # sim.data.qpos[model.joint_name2id('1')] += (initial_theta1_1 - sim.data.qpos[model.joint_name2id('1')])
        # sim.data.qpos[model.joint_name2id('2')] += (initial_theta2_1 - sim.data.qpos[model.joint_name2id('2')])
        # sim.data.qpos[model.joint_name2id('3')] += (initial_theta1_2 - sim.data.qpos[model.joint_name2id('3')])
        # sim.data.qpos[model.joint_name2id('4')] += (initial_theta2_2 - sim.data.qpos[model.joint_name2id('4')])
        # sim.forward()  # Recalculate positions based on new qpos values

    # sim.data.xfrc_applied[sim.model.body_name2id('object'), :] = force

    # Example parameters for Jacobian calculation for finger 1
    l1 = 1.732
    l2 = 1
    t1 = np.array([0, -1])
    n1 = np.array([1, 0])
    w1 = W(t1, n1)
    R1 = rotation_matrix_2d(0)
    J1 = jacobian(l1, l2, theta1_1, theta2_1)
    HJR1 = hand_jacobian(t1, n1, w1, R1, J1)

    # Example parameters for Jacobian calculation for finger 2
    t2 = np.array([0, 1])
    n2 = np.array([-1, 0])
    w2 = W(t2, n2)
    R2 = rotation_matrix_2d(0)
    J2 = jacobian(l1, l2, theta1_2, theta2_2)
    HJR2 = hand_jacobian(t2, n2, w2, R2, J2)

    # Adjusting HJR2 to the correct position in the full Jacobian
    O = np.zeros((2, 2))
    HJR1_full = np.concatenate([HJR1, O], axis=1)
    HJR2_full = np.concatenate([O, HJR2], axis=1)

    Hand_jacobian = np.concatenate([HJR1_full, HJR2_full], axis=0)
    Jh = Hand_jacobian

    # Calculate forces
    N = np.array([1, 1, 1, 1])
    G_pinv = np.linalg.pinv(G)

    f_control = G_pinv @ (Kp * (Xd - X))
    f_nullspace = (np.eye(4) - G_pinv @ G) @ N
    f = f_control + f_nullspace

    # Calculate joint torques
    
    Jh_pinv = np.linalg.pinv(Jh)
    joint_torques = Jh_pinv @ f

    # Apply torques to the actuators
    sim.data.ctrl[:] = joint_torques

    # Print relevant information every 100 steps
    if i % 100 == 0:
        print(f"Step: {i}")
        print(f"Position (X): {X}")
        print(f"Force (f): {f}")
        print(f"Joint Torques: {joint_torques}")
        print(Jh)

    # Step the simulation
    sim.step()

    # Render the simulation
    viewer.render()