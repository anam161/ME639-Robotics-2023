import cv2
import cv2.aruco as aruco
import numpy as np

# Load camera calibration parameters
camera_matrix = np.array([[542.72140565, 0, 307.76862441],
                          [0, 564.13186784, 224.63433001],
                          [0, 0, 1]])
dist_coeffs = np.array([-0.08048309, 0.59678798, -0.01890385, -0.02925722, -1.08808723])

# Open the camera
cap = cv2.VideoCapture(0)

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

# Main loop to detect markers and draw axes
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        # Draw markers
        aruco.drawDetectedMarkers(frame, corners, ids)
        
        # Estimate pose for each detected marker and draw the axes
        for i in range(len(ids)):
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[i], 0.1, camera_matrix, dist_coeffs)

            # Draw the axes for both origin (0,0,0) and the object
            frame = draw_origin_and_object_axes(frame, camera_matrix, dist_coeffs, rvec[0], tvec[0])
            

    # Display the frame with markers and axes
    cv2.imshow('ArUco Detection and Pose Estimation', frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
