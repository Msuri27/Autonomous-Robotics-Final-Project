"""student_controller controller."""

import math
import numpy as np

KP = 2.0
KI = 0.0
KD = 0.0

K_PATH = 0.05
MAX_CORR = 0.08

SET_SPEED = 6.5
MAX_SPEED = 6.5

ALPHA = 0.8

# cost function weights
W_PATH = 0.0
W_GOAL = 0.0
W_DISTANCE = 0.0

# mpc time horizon
TIME_HORIZON = 15
CONTACT_DISTANCE = 0.25

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

        # path
        self.desired_path_angle = 0.0
        self.desired_path_point = (0.0, 0.0)
        self.prev_e_path = 0.0

        self.right_speed = SET_SPEED
        self.left_speed = SET_SPEED

        self.first_step = True
        self.goal_coords = (0, 0)

        self.ds_candidates = [0.02, 0.04, 0.07 ]
        self.dtheta_candidates = [-0.20, -0.10, 0.0, 0.10, 0.20]

    def wrap(self, angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

    def ekf_prediction(self, ds, dtheta):
        x_prev = self.mu[0]
        y_prev = self.mu[1]
        theta_prev = self.mu[2]

        theta_new = self.wrap(theta_prev + dtheta)

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
                           theta_new
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
        phi_pred = self.wrap(phi_pred)

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
        self.mu[2] = self.wrap(self.mu[2])
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
            if landmark_id == "center_circle":
                # if somethings not seen but label is still in sensors ignore it
                try:
                    r_meas, phi_meas = sensors[landmark_id]
                    x_j, y_j = self.heuristic(landmark_id, r_meas, phi_meas)
                    self.ekf_update(x_j, y_j, r_meas, phi_meas)
                except:
                    pass
            elif landmark_id in ["goal", "penalty_cross", "corners"]:
                try:
                    for (r_meas, phi_meas) in sensors[landmark_id]:
                        x_j, y_j = self.heuristic(landmark_id, r_meas, phi_meas)
                        self.ekf_update(x_j, y_j, r_meas, phi_meas)
                except:
                    pass

    def pid_controller(self, sensors):
        r_ball, phi_ball = sensors["ball"]

        ball_bearing_world = self.mu[2] + phi_ball
        ball_y = self.mu[1] + r_ball * np.sin(ball_bearing_world)

        e_path = ball_y

        path_correction = K_PATH * e_path
        path_correction = max(-MAX_CORR, min(MAX_CORR, path_correction))

        desired_push_angle = self.desired_path_angle + path_correction
        desired_phi = self.wrap(desired_push_angle - self.mu[2])

        error = self.wrap(phi_ball - desired_phi)

        turn = KP * error
        turn = max(-0.20 * MAX_SPEED, min(0.20 * MAX_SPEED, turn))

        left_motor = SET_SPEED - turn
        right_motor = SET_SPEED + turn

        left_motor = max(-MAX_SPEED, min(MAX_SPEED, left_motor))
        right_motor = max(-MAX_SPEED, min(MAX_SPEED, right_motor))

        return {
            "left_motor": left_motor,
            "right_motor": right_motor
        }
    
    def mpc_predict(self, sensors, ds, dtheta):
        x_r, y_r, theta_r = self.mu
        r_ball, phi_ball = sensors["ball"]

        x_b = x_r + r_ball * np.cos(theta_r + phi_ball)
        y_b = y_r + r_ball * np.sin(theta_r + phi_ball)

        running_cost = 0.0
        counter = 0
        while(counter < TIME_HORIZON):
            x_r_next = x_r + ds * np.cos(theta_r)
            y_r_next = y_r + ds * np.sin(theta_r)
            theta_r_next = self.wrap(theta_r + dtheta)

            dist_to_ball = np.hypot(x_b - x_r, y_b - y_r)
            dist_to_ball_next = np.hypot(x_b - x_r_next, y_b - y_r_next)

            if min(dist_to_ball, dist_to_ball_next) < CONTACT_DISTANCE:
                x_b_next = x_b + ds * ALPHA * np.cos(theta_r)
                y_b_next = y_b + ds * ALPHA * np.sin(theta_r)
            else:
                x_b_next = x_b
                y_b_next = y_b

            running_cost += self.mpc_cost_function([x_r_next, y_r_next, theta_r_next, x_b_next, y_b_next])

            x_r = x_r_next
            y_r = y_r_next
            theta_r = theta_r_next
            x_b = x_b_next
            y_b = y_b_next

            counter += 1

        # return full state
        return running_cost
        
    def mpc_cost_function(self, predicted_state):
        x_r, y_r, theta_r, x_b, y_b = predicted_state
        
        return (W_PATH * (y_b)**2) + (W_GOAL * (self.goal_coords[0] - x_b)**2) + (W_DISTANCE * (np.hypot(x_r - x_b, y_r - y_b))**2)
    
    def mpc(self, sensors):
        all_running_costs = []

        for ds_candidate in self.ds_candidates:
            for dtheta_candidate in self.dtheta_candidates:
                cost = self.mpc_predict(sensors, ds_candidate, dtheta_candidate)
                all_running_costs.append((ds_candidate, dtheta_candidate, cost))
    
        lowest_cost = np.inf
        best_candidate = (0, 0)
        for candidate in all_running_costs:
            if candidate[2] < lowest_cost:
                lowest_cost = candidate[2]
                best_candidate = (candidate[0], candidate[1])
        
        return best_candidate

    def traj_gen(self, sensors):
        pass

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

        if self.first_step:
            sensors["goal"][0][0]
            self.goal_coords = self.map["goal"][0]       # whichever goal we're facing on startup should be the goal we want to score in (risky)
            self.first_step = False

        
        best_ds, best_dtheta = self.mpc(sensors)

        gain_ds = MAX_SPEED / max(self.ds_candidates)
        gain_dtheta = MAX_SPEED / max(self.dtheta_candidates)

        forward = gain_ds * best_ds
        turn = gain_dtheta * best_dtheta
    
        control_dict["left_motor"] = forward - turn
        control_dict["right_motor"] = forward + turn

        # control_dict = self.pid_controller(sensors)

        # if (1.5 < self.mu[2] < 1.6):
        #     control_dict["left_motor"] = 6.5
        #     control_dict["right_motor"] = 6.5

        print(sensors["goal"])

        return control_dict