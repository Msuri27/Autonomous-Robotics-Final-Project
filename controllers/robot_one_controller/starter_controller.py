"""student_controller controller."""

import math
import numpy as np

#constants
KP = 2.0
KI = 0.01
KD = 2.0
DESIRED_BALL_HEADING = 0.0
TIME_STEP = 32
MAX_SPEED = 6.5

class StudentController:
    def __init__(self):
        # ekf
        self.mu = np.array([0.0, 0.0, 0.0])
        self.Sigma = np.diag([0.1, 0.1, 1.0])

        # landmark map for feature matching
        self.map = {
            "center_circle": [(0,0)],
            "goal": [(4.5, 0), (-4.5, 0)],
            "penalty_cross": [(3.25, 0), (-3.25, 0)],
            "corners": [(-4.5, 3), (-4.5, -3), (4.5, 3), (4.5, -3)]
        }

        # pid integral error
        self.integral_error = 0.0
        self.previous_error = 0.0

    def ekf_prediction(self, ds, dtheta):
        x_prev = self.mu[0]
        y_prev = self.mu[1]
        theta_prev = self.mu[2]

        theta = theta_prev + dtheta

        # print("theta_prev:", theta_prev)
        # print("pred dx step:", ds*np.cos(theta_prev), ds*np.sin(theta_prev))

        # jacobian of a differential drive motion model (called A in the math i read)
        A = np.array([[1, 0, -ds*np.sin(theta_prev)],
                        [0, 1, ds*np.cos(theta_prev)], 
                        [0, 0, 1]
        ])

        # mean state as predicted by motion model
        mu_bar = np.array([x_prev + ds*np.cos(theta_prev), 
                        y_prev + ds*np.sin(theta_prev),
                        np.arctan2(np.sin(theta), np.cos(theta))
        ])

        # simple noise, tune with velocity and coefficients later
        Q = np.diag([ 
            (0.1 * ds + 0.01)**2, 
            (0.1 * ds + 0.01)**2, 
            (0.1 * abs(dtheta) + 0.01)**2 
        ])

        # compute covariance matrix
        Sigma_bar = A @ self.Sigma @ A.T + Q

        # update our global sigma and mu
        self.mu = mu_bar
        self.Sigma = Sigma_bar

    def ekf_update(self, x_j, y_j, r_meas, phi_meas):
        z = np.array([r_meas, phi_meas])

        dx = x_j - self.mu[0]
        dy = y_j - self.mu[1]

        q = dx**2 + dy**2

        r_pred = np.sqrt(q)
        phi_pred = np.arctan2(dy, dx) - self.mu[2]
        phi_pred = np.arctan2(np.sin(phi_pred), np.cos(phi_pred))

        z_hat = np.array([r_pred, phi_pred])

        # observation model jacobian
        H = np.array([[-dx/r_pred, -dy/r_pred, 0],
                    [dy/q, -dx/q, -1]
        ])

        R = np.diag([0.1**2, 0.05**2])

        # innovation covariance
        S = H @ self.Sigma @ H.T + R

        # kalman gain
        K = self.Sigma @ H.T @ np.linalg.inv(S) 

        # measurement error also wrapping angle
        y = z - z_hat
        y[1] = np.arctan2(np.sin(y[1]), np.cos(y[1]))

        self.mu = self.mu + K @ y
        self.mu[2] = np.arctan2(np.sin(self.mu[2]), np.cos(self.mu[2]))
        self.Sigma = (np.identity(3) - (K @ H)) @ self.Sigma

        # return H, S, K, y

    # euclidean distance
    def heuristic(self,landmark_id, r_meas, phi_meas):
        true_landmark_coords = (0, 0)
        true_bearing = self.mu[2] + phi_meas
        
        x_meas = self.mu[0] + r_meas * np.cos(true_bearing)
        y_meas = self.mu[1] + r_meas * np.sin(true_bearing)
        cartesian_coords = (x_meas, y_meas)
        
        min_dist = np.inf
        for landmark_coords in self.map[landmark_id]:
            dist = np.sqrt((cartesian_coords[0] - landmark_coords[0])**2 + (cartesian_coords[1] - landmark_coords[1])**2)

            if dist < min_dist:
                min_dist = dist
                true_landmark_coords = landmark_coords

        return true_landmark_coords
    
    # landmark matching using heuristic (euclidean)
    def feature_match(self, sensors):
        for landmark_id in sensors:
            if landmark_id == "center_cirlce":
                r_meas, phi_meas = sensors[landmark_id]
                (x_j, y_j) = self.heuristic(landmark_id, r_meas, phi_meas)
                self.ekf_update(x_j, y_j, r_meas, phi_meas)
            elif landmark_id in ["goal", "penalty_cross", "corners"]:
                for (r_meas, phi_meas) in sensors[landmark_id]:
                    (x_j, y_j) = self.heuristic(landmark_id, r_meas, phi_meas)
                    self.ekf_update(x_j, y_j, r_meas, phi_meas)
        
    def step(self, sensors):
        """
        Compute robot control as a function of sensors.

        Input:
        sensors: dict, contains current sensor values.

        Output:
        control_dict: dict, contains control for "left_motor" and "right_motor"
        """
        control_dict = {"left_motor": 0.0, "right_motor": 0.0}

        ds, dtheta = sensors["odometry"]

        self.ekf_prediction(ds, dtheta)
        self.feature_match(sensors)

        estimated_pose = self.mu.tolist()
        print(estimated_pose)
        
        meas_ball_heading = sensors["ball"][1]
        error = DESIRED_BALL_HEADING - meas_ball_heading

        # p stuff
        p_term = KP * error

        # i stuff
        dt = TIME_STEP / 1000.0
        self.integral_error += error * dt
        self.integral_error = max(-0.5, min(0.5, self.integral_error))      # bound error over time
        i_term = KI * self.integral_error

        # d stuff
        derivative = (error - self.previous_error) / dt
        d_term = KD * derivative
        self.previous_error = error

        turn = p_term + i_term + d_term
        turn = max(-0.5 * MAX_SPEED, min(0.5 * MAX_SPEED, turn)) # another stupid check

        left_motor = MAX_SPEED + turn
        right_motor = MAX_SPEED - turn

        # speed clipping
        control_dict["left_motor"] = max(-MAX_SPEED, min(MAX_SPEED, left_motor))
        control_dict["right_motor"] = max(-MAX_SPEED, min(MAX_SPEED, right_motor))

        # if (1.5 < self.mu[2] < 1.6):
        #     control_dict["left_motor"] = 6.5
        #     control_dict["right_motor"] = 6.5

        print(sensors["goal"])

        return control_dict
