import numpy as np
import time
import cv2
import cv2.aruco as aruco
from leap_hand_utils.dynamixel_client import *
import leap_hand_utils.leap_hand_utils as lhu

# Camera and ArUco setup
camera_matrix = np.array([[542.72140565, 0, 307.76862441],
                          [0, 564.13186784, 224.63433001],
                          [0, 0, 1]])

dist_coeffs = np.array([-0.08048309, 0.59678798, -0.01890385, -0.02925722, -1.08808723])

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters_create()

cap = cv2.VideoCapture(1)

# Define the ArUco dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# Define the ArUco detector parameters
parameters = aruco.DetectorParameters_create()

# Custom function to draw axes on the frame
def draw_axis(frame, camera_matrix, dist_coeffs, rvec, tvec, length=0.1):
    axis = np.float32([[length,0,0], [0,length,0], [0,0,-length], [0,0,0]]).reshape(-1,3)
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)
    
    # Convert points to integer tuples
    corner = tuple(map(int, imgpts[3].ravel()))
    pt1 = tuple(map(int, imgpts[0].ravel()))
    pt2 = tuple(map(int, imgpts[1].ravel()))
    pt3 = tuple(map(int, imgpts[2].ravel()))

    # Draw the axes lines
    frame = cv2.line(frame, corner, pt1, (255,0,0), 5)  # X-axis in red
    frame = cv2.line(frame, corner, pt2, (0,255,0), 5)  # Y-axis in green
    frame = cv2.line(frame, corner, pt3, (0,0,255), 5)  # Z-axis in blue
    return frame

# Function to calculate relative pose between two frames
def calculate_relative_pose(rvec1, tvec1, rvec2, tvec2):
    # Convert both rvecs into rotation matrices
    R1, _ = cv2.Rodrigues(rvec1)
    R2, _ = cv2.Rodrigues(rvec2)
    
    # Ensure the translation vectors are correctly shaped
    tvec1 = tvec1.reshape(3, 1)
    tvec2 = tvec2.reshape(3, 1)
    
    # Relative translation vector
    t_relative = tvec2 - tvec1
    
    # Apply a correction for the Z-axis
    t_relative[2] -= 0.24  # Subtract 1 cm (adjust as needed) from the Z-axis to correct the drift
    
    # Relative rotation matrix
    R_relative = R2 @ R1.T  # R_relative = R2 * R1^T (rotation of frame 2 relative to frame 1)
    
    # Convert relative rotation matrix back to a rotation vector
    r_relative, _ = cv2.Rodrigues(R_relative)
    
    return r_relative, t_relative



# Function to draw both origin and object axes with the object frame orientation set to zero
def draw_origin_and_object_axes(frame, camera_matrix, dist_coeffs, rvec_object, tvec_object):
    # Fixed origin frame (0,0,0 position and no rotation)
    origin_rvec = np.array([0, 0, 0], dtype=np.float32)  # No rotation
    origin_tvec = np.array([0, 0, 0], dtype=np.float32)  # No translation

    # Override the object's rotation vector to zero (no rotation)
    rvec_object_zero = np.array([0, 0, 0], dtype=np.float32)

    # Draw the fixed origin axis
    frame = draw_axis(frame, camera_matrix, dist_coeffs, origin_rvec, origin_tvec, 0.1)

    # Draw the object axis with zero orientation (aligned with origin orientation)
    frame = draw_axis(frame, camera_matrix, dist_coeffs, rvec_object_zero, tvec_object, 0.1)

    # Calculate relative pose between origin and object
    rvec_relative, tvec_relative = calculate_relative_pose(origin_rvec, origin_tvec, rvec_object_zero, tvec_object)

    # Concatenate tvec_relative and set the rvec_relative (orientation) to zero
    X = np.hstack((tvec_relative.ravel(), np.zeros(3)))  # Orientation is forced to zero
    
    # Print the object pose with orientation as zero
    print("X:", X)

    # Check if the object frame coincides with the origin frame
    if np.allclose(tvec_relative, np.zeros(3), atol=1e-2):
        print("Frames coincide: [0, 0, 0, 0, 0, 0]")

    return frame

# Impedance control functions
def force_vec(s, t, n):
    z = np.column_stack([s, t, n])
    return z

s1 = np.array([0, 1, 0])
t1 = np.array([0, 0, 1])
n1 = np.array([1, 0, 0])
z1 = force_vec(s1, t1, n1)

s2 = np.array([0, 1, 0])
t2 = np.array([0, 0, -1])
n2 = np.array([-1, 0, 0])
z2 = force_vec(s2, t2, n2)

# Function to calculate cross product
def cross_prod(x, vec):
    return np.cross(x, vec)

x1 = np.array([-0.03, 0, 0])
p1 = cross_prod(x1, s1)
q1 = cross_prod(x1, t1)
r1 = cross_prod(x1, n1)
k = np.column_stack([p1, q1, r1])
G1 = np.concatenate([z1, k])

x2 = np.array([0.03, 0, 0])
p2 = cross_prod(x2, s2)
q2 = cross_prod(x2, t2)
r2 = cross_prod(x2, n2)
l = np.column_stack([p2, q2, r2])

G2 = np.concatenate([z2, l])
G = np.block([G1, G2])

def W(s, t, n):
    s = np.array(s).reshape(3, 1)
    t = np.array(t).reshape(3, 1)
    n = np.array(n).reshape(3, 1)
    w = np.hstack([s, t, n])
    return w

def rotation_matrix_3d(theta):
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])
    return R

def jacobian(l1, l2, l3, theta1, theta2, theta3):
    J = np.array([
        [-l1*np.sin(theta1) - l2*np.sin(theta1+theta2) - l3*np.sin(theta1+theta2+theta3),
         -l2*np.sin(theta1+theta2) - l3*np.sin(theta1+theta2+theta3),
         -l3*np.sin(theta1+theta2+theta3)],
        [l1*np.cos(theta1) + l2*np.cos(theta1+theta2) + l3*np.cos(theta1+theta2+theta3),
         l2*np.cos(theta1+theta2) + l3*np.cos(theta1+theta2+theta3),
         l3*np.cos(theta1+theta2+theta3)],
        [0, 0, 0] 
    ])
    return J

def hand_jacobian(s, t, n, w, R, J):
    WRJ = np.dot(np.dot(w.T, R), J)
    return WRJ

# Finger 1 parameters
l1_f1, l2_f1, l3_f1 = 0.05, 0.035, 0.05
theta1_f1, theta2_f1, theta3_f1 = 0.942, 0.44, 0.22

s1 = np.array([0, 1, 0])
t1 = np.array([0, 0, 1])
n1 = np.array([1, 0, 0])
w1 = W(s1, t1, n1)
R1 = rotation_matrix_3d(0)
J1 = jacobian(l1_f1, l2_f1, l3_f1, theta1_f1, theta2_f1, theta3_f1)
HJR1 = hand_jacobian(s1, t1, n1, w1, R1, J1)

# Finger 2 parameters
l1_f2, l2_f2, l3_f2 = 0.05, 0.035, 0.05
theta1_f2, theta2_f2, theta3_f2 = 1.54, 0.251, 0.0942

s2 = np.array([0, 1, 0])
t2 = np.array([0, 0, -1])
n2 = np.array([-1, 0, 0])
w2 = W(s2, t2, n2)
R2 = rotation_matrix_3d(0)
J2 = jacobian(l1_f2, l2_f2, l3_f2, theta1_f2, theta2_f2, theta3_f2)
HJR2 = hand_jacobian(s2, t2, n2, w2, R2, J2)

O = np.zeros((3, 3))
HJR1_full = np.concatenate([HJR1, O], axis=1)
HJR2_full = np.concatenate([O, HJR2], axis=1)
Hand_jacobian = np.concatenate([HJR1_full, HJR2_full], axis=0)
Jh = Hand_jacobian

G_pinv = np.linalg.pinv(G)

Xd = np.array([0, 0, 0, 0, 0, 0])  # Desired pose
Kp = np.array([[0.1,0,0,0,0,0],
              [0,0.1,0,0,0,0],
              [0,0,0.1,0,0,0],
              [0,0,0,0.1,0,0],
              [0,0,0,0,0.1,0],
              [0,0,0,0,0,0.1]])  # Proportional gain matrix
N = np.array([0, 0, 0, 0, 0, 0])  # Nullspace forces



# Hardware control class
class LeapNode:
    def __init__(self):
        self.torque_lower = -1023
        self.torque_upper = 1023
        self.kP = 500
        self.kI = 0
        self.kD = 300
        self.curr_lim = 350
        self.motors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        self.num_motors = len(self.motors)
        try:
            self.dxl_client = DynamixelClient(self.motors, '/dev/ttyUSB0', 4000000)
            self.dxl_client.connect()
        except Exception:
            self.dxl_client = DynamixelClient(self.motors, 'COM10', 4000000)
            self.dxl_client.connect()

        self.dxl_client.sync_write(self.motors, [0] * self.num_motors, 11, 1)
        time.sleep(0.5)
        self.dxl_client.set_torque_enabled(self.motors, True)
        self.dxl_client.sync_write(self.motors, np.ones(self.num_motors) * self.curr_lim, 102, 2)

    def calculate_and_set_torque(self, tau_initial):
        torque_commands = np.zeros(self.num_motors)
        torque_commands[:6] = tau_initial[:6]
        torque_command1 = np.copy(torque_commands)
        torque_command1[3] = -abs(torque_command1[3])
        torque_command1[:3] = np.abs(torque_command1[:3])
        self.set_torque(torque_command1)
        print(f"Torque commands: {torque_command1}")


    def set_torque(self, torque_values):
        if len(torque_values) != self.num_motors:
            raise ValueError(f"Expected {self.num_motors} torque values, but got {len(torque_values)}")
        actual_torque = lhu.apply_torque_to_LEAPhand(torque_values, self.torque_lower, self.torque_upper)
        self.cur = np.array(actual_torque)
        print(f"Setting torques: {self.cur} for motors: {self.motors}")
        self.dxl_client.write_desired_cur(self.motors, self.cur)

    def read_pos(self):
        return self.dxl_client.read_pos()

    def read_cur(self):
        return self.dxl_client.read_cur()

# Real-time update of X from pose estimation
def main():
    leap_hand = LeapNode()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            for i in range(len(ids)):
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[i], 0.1, camera_matrix, dist_coeffs)
                
                # Assuming pose relative to origin is required
                
                X = np.hstack((tvec[0].ravel(), np.zeros(3)))  # Update current pose (position + zero orientation)
                print(X)

                # Calculate control forces
                f_control = G_pinv @ (Kp @ (Xd - X))
                f_nullspace = (np.eye(6) - G_pinv @ G) @ N
                f = f_control + f_nullspace

                # Compute torques and apply them
                tau_initial = Jh.T @ f
                leap_hand.calculate_and_set_torque(tau_initial)

        time.sleep(0.5)

if __name__ == "__main__":
    main()
