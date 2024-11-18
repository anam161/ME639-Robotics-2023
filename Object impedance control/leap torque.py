#!/usr/bin/env python3
import numpy as np
import time
from leap_hand_utils.dynamixel_client import *
import leap_hand_utils.leap_hand_utils as lhu

class LeapNode:
    def __init__(self):

        # Parameters
        self.torque_lower = -1023  # Example lower limit
        self.torque_upper = 1023   # Example upper limit

        #### Some parameters
        self.kP = 600
        self.kI = 0
        self.kD = 200
        self.curr_lim = 350

        # Initialize current (torque) values
        self.cur = lhu.apply_torque_to_LEAPhand(np.zeros(16), self.torque_lower, self.torque_upper)

        # Initialize motors
        self.motors = motors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        try:
            self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB0', 4000000)
            self.dxl_client.connect()
        except Exception:
            try:
                self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB1', 4000000)
                self.dxl_client.connect()
            except Exception:
                self.dxl_client = DynamixelClient(motors, 'COM10', 4000000)
                self.dxl_client.connect()

        # Set operating mode to torque control
        self.dxl_client.sync_write(motors, [0] * len(motors), 11, 1)  # Set to torque control mode
        time.sleep(0.5)  # Delay to ensure the mode is set

        # Enable torque
        self.dxl_client.set_torque_enabled(motors, True)

        # Set current limit
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.curr_lim, 102, 2)  # Set current limit

    # Method to set torque
    def set_torque(self, torque_values):
        # Convert torque values to actual range before sending
        actual_torque = lhu.apply_torque_to_LEAPhand(torque_values, self.torque_lower, self.torque_upper)
        self.cur = np.array(actual_torque)
        self.dxl_client.write_desired_cur(self.motors, self.cur)

    # Read position
    def read_pos(self):
        return self.dxl_client.read_pos()

    # Read current
    def read_cur(self):
        return self.dxl_client.read_cur()

# Initialize the node
def main(**kwargs):
    leap_hand = LeapNode()
    while True:
        torque_commands = np.array([0, 0.05, 0.03, 0.03, 0, 0.05, 0.03, 0.03, 0, 0, 0, 0, 0, 0, 0.07, 0.07])  # Example torque values
        leap_hand.set_torque(torque_commands)
        print("Current: " + str(leap_hand.read_cur()))
        time.sleep(0.5)

if __name__ == "__main__":
    main()
