# -*- coding: utf-8 -*-
"""Untitled28.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1JMQMkomLZs47u3O5paLcQ5QcaBPf9x09
"""

import numpy as np

def grasp_matrix_2d(contact_points):
    """
    Constructs a grasp matrix for multiple 2D contact points.

    Parameters:
    contact_points (list of tuples): Each tuple contains (t, n, x) where
        t (np.array): Tangential direction vector at contact point.
        n (np.array): Normal direction vector at contact point.
        x (np.array): Position vector of the contact point.

    Returns:
    np.array: The grasp matrix G.
    """
    G_list = []

    for t, n, x in contact_points:
        t = np.atleast_2d(t).T
        n = np.atleast_2d(n).T
        x = np.atleast_2d(x).T

        G_i = np.block([[t, n], [np.cross(x.T, t.T).T, np.cross(x.T, n.T).T]])
        G_list.append(G_i)

    G = np.hstack(G_list)
    return G

# Test (value of (t,n,x))
contact_points = [
    (np.array([0, -1]), np.array([1, 0]), np.array([-2, 0])),
    (np.array([0, -1]), np.array([1, 0]), np.array([2, 0]))
]

G = grasp_matrix_2d(contact_points)
print(G)

# calculate the finger force
Kp=np.array([20,20,20])
Xd = np.array([0,0.2,0])
X = np.array([0,0,0])
Kd = np.array([20,20,20])
Vd = np.array([0,0.1,0])
V = np.array([0,0,0])

f = G.T @ (Kp*(Xd-X)+(Kd*(Vd-V)))
print(f)