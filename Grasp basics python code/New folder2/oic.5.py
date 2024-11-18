import cv2
import cv2.aruco as aruco
import numpy as np
import time
from leap_hand_utils.dynamixel_client import *
import leap_hand_utils.leap_hand_utils as lhu

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

x1 = np.array([-0.02, 0, 0])
p1 = cross_prod(x1, s1)
q1 = cross_prod(x1, t1)
r1 = cross_prod(x1, n1)
k = np.column_stack([p1, q1, r1])
G1 = np.concatenate([z1, k])

x2 = np.array([0.02, 0, 0])
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
theta1_f1, theta2_f1, theta3_f1 = 4.75, 3.5,  3.14

s1 = np.array([0, 1, 0])
t1 = np.array([0, 0, 1])
n1 = np.array([1, 0, 0])
w1 = W(s1, t1, n1)
R1 = rotation_matrix_3d(0)
J1 = jacobian(l1_f1, l2_f1, l3_f1, theta1_f1, theta2_f1, theta3_f1)
HJR1 = hand_jacobian(s1, t1, n1, w1, R1, J1)

# Finger 2 parameters
l1_f2, l2_f2, l3_f2 = 0.05, 0.035, 0.05
theta1_f2, theta2_f2, theta3_f2 = 1.608, 3.5,  3.14

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

# Load camera calibration parameters
camera_matrix = np.array([[542.72140565, 0, 307.76862441],
                          [0, 564.13186784, 224.63433001],
                          [0, 0, 1]])
dist_coeffs = np.array([-0.08048309, 0.59678798, -0.01890385, -0.02925722, -1.08808723])

# Open the camera
cap = cv2.VideoCapture(0)

# Define the ArUco dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters_create()

# Global pose (X) variable that will be shared between ArUco detection and torque control
X = np.zeros(6)  # This will store [tx, ty, tz, 0, 0, 0] for real-time updates

# Custom function to draw axes on the frame
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
    R1, _ = cv2.Rodrigues(rvec1)
    R2, _ = cv2.Rodrigues(rvec2)
    tvec1 = tvec1.reshape(3, 1)
    tvec2 = tvec2.reshape(3, 1)
    t_relative = tvec2 - tvec1
    t_relative[2] -= 0.5 # Apply correction for Z drift
    R_relative = R2 @ R1.T
    r_relative, _ = cv2.Rodrigues(R_relative)
    return r_relative, t_relative

# Palm frame transformation
def get_palm_frame_transformation():
    # Define a fixed transformation from object frame to palm frame (above the object)
    translation_vector = np.array([0, 0, 0.145])  # Palm is 14.5cm above the object
    rotation_matrix = np.eye(3)  # Identity matrix (no rotation between frames)
    return translation_vector, rotation_matrix

# Draw object axes and palm frame axes relative to each other
def draw_origin_and_object_axes(frame, camera_matrix, dist_coeffs, rvec_object, tvec_object):
    global X  # Use global X variable to update it

    # Get palm frame transformation (translation and rotation)
    palm_translation, palm_rotation = get_palm_frame_transformation()

    # Convert rvec_object to a rotation matrix
    R_object, _ = cv2.Rodrigues(rvec_object)

    # Apply palm frame transformation to the object's position
    tvec_palm_frame = palm_rotation @ tvec_object.T + palm_translation.reshape(-1, 1)
    tvec_palm_frame = tvec_palm_frame.T.flatten()  # Convert back to a 1D array

    # Draw the origin and palm axes
    origin_rvec = np.array([0, 0, 0], dtype=np.float32)
    origin_tvec = np.array([0, 0, 0], dtype=np.float32)
    rvec_palm = np.array([0, 0, 0], dtype=np.float32)  # Assuming palm and object are aligned

    frame = draw_axis(frame, camera_matrix, dist_coeffs, origin_rvec, origin_tvec, 0.1)  # Origin axes
    frame = draw_axis(frame, camera_matrix, dist_coeffs, rvec_palm, tvec_palm_frame, 0.1)  # Palm axes

    # Calculate relative pose between origin and palm frame
    rvec_relative, tvec_relative = calculate_relative_pose(origin_rvec, origin_tvec, rvec_palm, tvec_palm_frame)

    # Update global X variable with the relative position (and keep zero orientation for simplicity)
    X = np.hstack((tvec_relative.ravel(), np.zeros(3)))  # Position relative to palm, orientation zeros for simplicity

    print(f"Real-time X relative to palm frame: {X}")  # Print the real-time X value
    
    return frame

# Function to run ArUco marker detection and update X in real-time
def aruco_detection_loop():
    global X
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            for i in range(len(ids)):
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[i], 0.1, camera_matrix, dist_coeffs)
                frame = draw_origin_and_object_axes(frame, camera_matrix, dist_coeffs, rvec[0], tvec[0])

        # Display the frame with markers and axes
        cv2.imshow('ArUco Detection and Pose Estimation', frame)

        # Break the loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Hardware control code using the real-time X value
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

    def calculate_and_set_torque(self):
        global X  # Use the real-time X value from the ArUco detection
        G_pinv = np.linalg.pinv(G)  # Assuming G is already computed
        
        # Desired position (Xd) - this is where you want the object to return
        Xd = np.zeros(6)  # This can be updated based on the target pose
        
        # Gains for PD control (spring-damper system)
        Kp = np.eye(6) * 10   # Proportional gain, stiff spring to return to desired position
        Kd = np.eye(6) * 5  # Damping gain, smoothens the return
        
        # Calculate the velocity of the object (assuming small time steps, Xdot ~ deltaX)
        Xdot = X - self.prev_X if hasattr(self, 'prev_X') else np.zeros(6)
    
        self.prev_X = X  # Store the previous position for next time step

        # Control force calculation (PD control)
        f_control = G_pinv @ (Kp @ (Xd - X)) #- Kd @ Xdot)  # Spring-damper control force
        
        # Nullspace forces (optional)
        N = np.ones(6) * 0.4  # Nullspace forces to maintain balance
        f_nullspace = (np.eye(6) - G_pinv @ G) @ N

        # Total force
        f = f_control + f_nullspace

        # Real-time joint angles
        joint_angles = self.read_pos()

        # Update hand Jacobians based on real-time joint angles for finger 1 and 2
        theta1_f1, theta2_f1, theta3_f1 = joint_angles[:3]  # Joint angles for finger 1
        J1 = jacobian(l1_f1, l2_f1, l3_f1, theta1_f1, theta2_f1, theta3_f1)
        HJR1 = hand_jacobian(s1, t1, n1, w1, R1, J1)

        theta1_f2, theta2_f2, theta3_f2 = joint_angles[3:6]  # Joint angles for finger 2
        J2 = jacobian(l1_f2, l2_f2, l3_f2, theta1_f2, theta2_f2, theta3_f2)
        HJR2 = hand_jacobian(s2, t2, n2, w2, R2, J2)

        # Concatenate the updated hand Jacobians
        HJR1_full = np.concatenate([HJR1, O], axis=1)
        HJR2_full = np.concatenate([O, HJR2], axis=1)
        Jh = np.concatenate([HJR1_full, HJR2_full], axis=0)  # Updated overall hand Jacobian

        # Compute the updated torque using the updated Jacobian
        tau_initial = Jh.T @ f  # Updated torque using the updated Jacobian

        # Print the real-time torque values
        print(f"Real-time torque (tau_initial): {tau_initial[:6]}")

        # Apply torque to the motors
        torque_commands = np.zeros(self.num_motors)
        torque_commands[:6] = tau_initial[:6]

        torque_command1 = np.copy(torque_commands)
        torque_command1[3] = -abs(torque_command1[3])
        torque_command1[:3] = np.abs(torque_command1[:3])
        self.set_torque(torque_command1)

    def set_torque(self, torque_values):
        actual_torque = lhu.apply_torque_to_LEAPhand(torque_values, self.torque_lower, self.torque_upper)
        self.cur = np.array(actual_torque)
        print(f"Setting torques: {self.cur}")
        self.dxl_client.write_desired_cur(self.motors, self.cur)

    def read_pos(self):
        return self.dxl_client.read_pos()

    def read_cur(self):
        return self.dxl_client.read_cur()


# Function to run torque control loop
def torque_control_loop():
    leap_hand = LeapNode()
    while True:
        leap_hand.calculate_and_set_torque()
        time.sleep(0.5)

# Main function to run both loops in parallel
if __name__ == "__main__":
    import threading

    # Create and start threads for both loops
    aruco_thread = threading.Thread(target=aruco_detection_loop)
    torque_thread = threading.Thread(target=torque_control_loop)

    aruco_thread.start()
    torque_thread.start()

    aruco_thread.join()
    torque_thread.join()
