import numpy as np
import time
import cv2
import cv2.aruco as aruco
from leap_hand_utils.dynamixel_client import *
import leap_hand_utils.leap_hand_utils as lhu

# ------------------------------- Object Impedance Control Code -------------------------------

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

# ------------------------------- Hardware Control Code -------------------------------

class LeapNode:
    def __init__(self):
        self.torque_lower = -1023
        self.torque_upper = 1023

        # Control parameters
        self.kP = 500
        self.kI = 0
        self.kD = 300
        self.curr_lim = 350

        # Motor directions (defined for 16 motors)
        self.motor_directions = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])

        # Initialize motors
        self.motors = [i for i in range(16)]
        self.num_motors = len(self.motors)

        try:
            self.dxl_client = DynamixelClient(self.motors, '/dev/ttyUSB0', 4000000)
            self.dxl_client.connect()
        except Exception:
            try:
                self.dxl_client = DynamixelClient(self.motors, '/dev/ttyUSB1', 4000000)
                self.dxl_client.connect()
            except Exception:
                self.dxl_client = DynamixelClient(self.motors, 'COM10', 4000000)
                self.dxl_client.connect()

        # Set operating mode to torque control
        self.dxl_client.sync_write(self.motors, [0] * self.num_motors, 11, 1)
        time.sleep(0.5)

        # Enable torque
        self.dxl_client.set_torque_enabled(self.motors, True)

        # Set current limit
        self.dxl_client.sync_write(self.motors, np.ones(self.num_motors) * self.curr_lim, 102, 2)

    def calculate_and_set_torque(self, tau_initial):
        # Ensure torque commands are within range
        torque_commands = np.zeros(self.num_motors) 
        torque_commands[:6] = tau_initial[:6]

        torque_command1 = np.copy(torque_commands)
        torque_command1[3] = -abs(torque_command1[3])
        torque_command1[:3] = np.abs(torque_command1[:3])

        self.set_torque(torque_command1)

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

# ------------------------------- Camera-Based Pose Estimation -------------------------------

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

def pose_estimation(rvec, tvec):
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    rotation_matrix_flat = rotation_matrix.flatten()
    pose_array = np.hstack((tvec.flatten(), rotation_matrix_flat[:3]))  # Use only first 3 values
    return pose_array

def draw_origin_and_object_axes(frame, camera_matrix, dist_coeffs, rvec, tvec):
    origin_rvec = np.array([0, 0, 0], dtype=np.float32)
    origin_tvec = np.array([0, 0, 0], dtype=np.float32)

    frame = draw_axis(frame, camera_matrix, dist_coeffs, origin_rvec, origin_tvec, 0.1)
    frame = draw_axis(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

    return frame

# ------------------------------- Main Loop (Impedance + Hardware + Vision) -------------------------------

def main():
    # Initialize Leap hand control
    leap_hand = LeapNode()

    # Camera setup for pose estimation
    camera_matrix = np.array([[542.72140565,   0,         307.76862441],
                              [  0,         564.13186784, 224.63433001],
                              [  0,           0,           1        ]])
    dist_coeffs = np.array([-0.08048309,  0.59678798, -0.01890385, -0.02925722, -1.08808723])

    cap = cv2.VideoCapture(0)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters_create()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)

            for i in range(len(ids)):
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[i], 0.1, camera_matrix, dist_coeffs)
                rvec = np.array(rvec, dtype=np.float32)
                tvec = np.array(tvec, dtype=np.float32)

                frame = draw_origin_and_object_axes(frame, camera_matrix, dist_coeffs, rvec[0], tvec[0])
                object_pose_array = pose_estimation(tvec[0],rvec[0])

                # Update impedance control variables with real-time pose values
                X = np.array([0,0,0,0,0,0])  # Extract X (position + orientation)

                # Object impedance control logic (tau_initial calculation)
                Kp = np.eye(6) * 10
                Xd = np.array([0, 0, 0, 0, 0, 0])  # Desired state
                N = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
                G = np.eye(6)  # Grasp matrix (you can modify this based on your simulation setup)

                G_pinv = np.linalg.pinv(G)
                f_control = G_pinv @ (Kp @ (Xd - X))  # Calculate control force
                f_nullspace = (np.eye(6) - G_pinv @ G) @ N
                f = f_control + f_nullspace

                Jh = np.eye(6)  # Hand Jacobian (you can modify this based on your simulation setup)
                tau_initial = Jh.T @ f

                # Now set the torques to the motors based on the calculated tau_initial
                leap_hand.calculate_and_set_torque(tau_initial)

        cv2.imshow('ArUco Detection and Pose Estimation', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
