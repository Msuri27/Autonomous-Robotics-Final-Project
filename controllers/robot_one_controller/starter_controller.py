"""student_controller controller."""

import math
import numpy as np

KP = 2.0
KI = 0.0
KD = 0.0

KP_BALL = 2.0

K_PATH = 0.05
MAX_CORR = 0.08

SET_SPEED = 6.5
MAX_SPEED = 6.5

ALPHA = 0.8

W_PATH = 500.0
W_GOAL = 0.1
W_CONTACT = 0.5
W_BEARING = 30.0
K_BALL_PATH = 2.0

TIME_HORIZON = 20
APPROACH_DISTANCE = 0.6
CONTACT_DISTANCE = 0.12
CONTACT_OFFSET = 0.3   # distance behind ball (tune 0.25–0.4)

STAGING_OFFSET = 0.25
STAGING_TOL = 0.35
ANGLE_TOL = 0.08
BALL_CENTER_TOL = 0.12
GOAL_REACHED_DIST = 0.25

KP_STAGE_HEADING = 2.5
KP_ALIGN = 2.0
KP_PUSH = 2.0

STAGE_SPEED = 3.0
ALIGN_SPEED = 0.5
PUSH_SPEED = 6.5

class StudentController:
    def __init__(self):
        self.finite_state = "FIND_BALL"
        
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
        self.goal_coords = (4.5, 0)       # NOTE: this must be changed later because the goal could be the other goal in 1v1

        self.ds_candidates = [0.05, 0.07, 0.10]
        self.dtheta_candidates = [-0.25, -0.12, 0.0, 0.12, 0.25]

        self.global_r_ball = 0
        self.global_phi_ball = 0

    def wrap(self, angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

    def ekf_prediction(self, sensors):
        ds, dtheta = sensors["odometry"]

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
                x_b_next = x_b + ds * ALPHA * np.cos(theta_r_next)
                y_b_next = y_b + ds * ALPHA * np.sin(theta_r_next)
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

        # path term (keep ball on y = 0), change to different trajectory later
        path_cost = W_PATH * (y_b ** 2)

        # goal term (push ball toward +x)
        goal_cost = W_GOAL * (self.goal_coords[0] - x_b) ** 2

        # contact geometry term (robot behind ball)
        # desired push direction (for now: straight toward goal)
        push_angle = 0.0

        # where robot should be relative to ball
        angle_to_ball = np.arctan2(y_b - y_r, x_b - x_r)
        phi_pred = self.wrap(angle_to_ball - theta_r)

        desired_phi = -K_BALL_PATH * y_b
        desired_phi = max(-0.35, min(0.35, desired_phi))

        bearing_cost = W_BEARING * (self.wrap(phi_pred - desired_phi) ** 2)

        return path_cost + goal_cost + bearing_cost
    
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
            # print(candidate)
        
        return best_candidate

    def find_ball(self, sensors):
        try:
            r_ball, phi_ball = sensors["ball"]
            self.finite_state = "GO_TO_STAGING_POINT"
            return {"left_motor": 0.0, "right_motor": 0.0}
        except:
            left_motor = -6.5
            right_motor = 6.5

            return {
                "left_motor": left_motor,
                "right_motor": right_motor
            }

    
    def get_ball_global(self, sensors):
        x_r, y_r, theta_r = self.mu

        try:
            r_ball, phi_ball = sensors["ball"]

            self.global_r_ball = r_ball
            self.global_phi_ball = phi_ball
        except:
            pass

        bearing = theta_r + self.global_phi_ball
        x_b = x_r + self.global_r_ball * np.cos(bearing)
        y_b = y_r + self.global_r_ball * np.sin(bearing)

        return x_b, y_b


    def get_line_of_action(self, sensors):
        x_b, y_b = self.get_ball_global(sensors)

        ball = np.array([x_b, y_b])
        goal = np.array(self.goal_coords)

        line_vec = goal - ball
        norm = np.linalg.norm(line_vec)

        if norm < 1e-6:
            line_dir = np.array([1.0, 0.0])
        else:
            line_dir = line_vec / norm

        staging = ball - STAGING_OFFSET * line_dir
        line_angle = np.arctan2(line_dir[1], line_dir[0])

        return x_b, y_b, staging, line_angle

    def go_to_staging_point(self, sensors):
        x_b, y_b, staging, line_angle = self.get_line_of_action(sensors)

        x_r, y_r, theta_r = self.mu

        dx = staging[0] - x_r
        dy = staging[1] - y_r

        dist = np.hypot(dx, dy)
        target_angle = np.arctan2(dy, dx)
        heading_error = self.wrap(target_angle - theta_r)

        turn = KP_STAGE_HEADING * heading_error
        turn = max(-0.35 * MAX_SPEED, min(0.35 * MAX_SPEED, turn))

        forward = STAGE_SPEED

        if dist < STAGING_TOL:
            self.finite_state = "ALIGN_TO_LINE"

        left = forward - turn
        right = forward + turn

        return {
            "left_motor": max(-MAX_SPEED, min(MAX_SPEED, left)),
            "right_motor": max(-MAX_SPEED, min(MAX_SPEED, right))
        }


    def align_to_line(self, sensors):
        x_b, y_b, staging, line_angle = self.get_line_of_action(sensors)

        theta_r = self.mu[2]

        heading_error = self.wrap(line_angle - theta_r)

        # Also keep ball centered while aligning
        error = heading_error + 0.5 * self.global_phi_ball

        turn = KP_ALIGN * error
        turn = max(-0.25 * MAX_SPEED, min(0.25 * MAX_SPEED, turn))

        forward = ALIGN_SPEED

        if abs(heading_error) < ANGLE_TOL and abs(self.global_phi_ball) < BALL_CENTER_TOL:
            self.finite_state = "PUSH_BALL"

        left = forward - turn
        right = forward + turn

        return {
            "left_motor": max(-MAX_SPEED, min(MAX_SPEED, left)),
            "right_motor": max(-MAX_SPEED, min(MAX_SPEED, right))
        }


    def push_ball(self, sensors):
        x_b, y_b, staging, line_angle = self.get_line_of_action(sensors)

        theta_r = self.mu[2]

        heading_error = self.wrap(line_angle - theta_r)

        # Primary: keep ball centered. Secondary: stay on line angle.
        error = self.global_phi_ball + 0.3 * heading_error

        turn = KP_PUSH * error
        turn = max(-0.15 * MAX_SPEED, min(0.15 * MAX_SPEED, turn))

        forward = PUSH_SPEED

        goal_dist = np.hypot(self.goal_coords[0] - x_b, self.goal_coords[1] - y_b)

        if goal_dist < GOAL_REACHED_DIST:
            self.finite_state = "DONE"

        # If ball gets badly off-center, recover by going back to staging
        if abs(self.global_phi_ball) > 0.45:
            self.finite_state = "GO_TO_STAGING_POINT"

        left = forward - turn
        right = forward + turn

        return {
            "left_motor": max(-MAX_SPEED, min(MAX_SPEED, left)),
            "right_motor": max(-MAX_SPEED, min(MAX_SPEED, right))
        }

    def step(self, sensors):
        """
        Compute robot control as a function of sensors.

        Input:
        sensors: dict, contains current sensor values.

        Output:
        control_dict: dict, contains control for "left_motor" and "right_motor"
        """
        control_dict = {"left_motor": 0.0, "right_motor": 0.0}

        self.ekf_prediction(sensors)
        self.feature_match(sensors)

        estimated_pose = self.mu.tolist()
        print(estimated_pose)

        print(self.finite_state)

        match self.finite_state:
            case "FIND_BALL":
                control_dict = self.find_ball(sensors)

            case "GO_TO_STAGING_POINT":
                control_dict = self.go_to_staging_point(sensors)

            case "ALIGN_TO_LINE":
                control_dict = self.align_to_line(sensors)

            case "PUSH_BALL":
                control_dict = self.push_ball(sensors)

            case "DONE":
                control_dict = {"left_motor": 0.0, "right_motor": 0.0}

        # control_dict = self.pid_controller(sensors)

        # if (1.5 < self.mu[2] < 1.6):
        #     control_dict["left_motor"] = 6.5
        #     control_dict["right_motor"] = 6.5

        return control_dict