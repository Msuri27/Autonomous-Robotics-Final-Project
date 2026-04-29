"""student_controller controller."""

import math
import numpy as np

MAX_SPEED = 6.5

# Line-of-action staging
LOA_OFFSET = 0.5       # distance behind ball toward opposite side of goal
STAGING_TOL = 0.2      # how close robot must get to offset point
FREEZE_DISTANCE = 0.45

# Turning thresholds
ANGLE_TOL = 0.06        # heading tolerance for turn-in-place states

# Goal/end condition
GOAL_REACHED_DIST = 0.25

# Speeds
TURN_SPEED = 2.0        # in-place turning speed
STAGE_SPEED = 2.0       # drive-to-offset speed
PUSH_SPEED = 5.0        # final push speed

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

        self.goal_coords = np.array([4.5, 0.0])       # NOTE: this must be changed later because the goal could be the other goal in 1v1

        self.ds_candidates = [0.05, 0.07, 0.10]
        self.dtheta_candidates = [-0.25, -0.12, 0.0, 0.12, 0.25]

        self.global_r_ball = 0
        self.global_phi_ball = 0

        self.prev_side_error = None
        self.drive_line_ball = None
        self.drive_line_unit = None
        self.drive_offset_point = None
        self.drive_line_angle = None
        self.initial_side_error = None

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

    def get_ball_polar_coords(self, sensors):
        ball_obs = sensors.get("ball", None)
        if ball_obs is not None:
            self.global_r_ball, self.global_phi_ball = ball_obs
        return self.global_r_ball, self.global_phi_ball


    def get_ball_coords(self, sensors):
        x_r, y_r, theta_r = self.mu
        r_ball, phi_ball = self.get_ball_polar_coords(sensors)

        bearing = theta_r + phi_ball
        return np.array([
            x_r + r_ball * np.cos(bearing),
            y_r + r_ball * np.sin(bearing)
        ])

    def get_loa_geometry(self, sensors):
        ball = self.get_ball_coords(sensors)
        goal = np.array(self.goal_coords)

        loa_vec = goal - ball
        norm = np.linalg.norm(loa_vec)

        if norm < 1e-6:
            loa_unit = np.array([1.0, 0.0])
        else:
            loa_unit = loa_vec / norm

        offset_point = ball - LOA_OFFSET * loa_unit
        loa_angle = np.arctan2(loa_unit[1], loa_unit[0])

        return ball, loa_unit, offset_point, loa_angle

    def find_ball(self, sensors):
        ball_obs = sensors.get("ball", None)

        if ball_obs is None:
            return {"left_motor": -MAX_SPEED, "right_motor": MAX_SPEED}

        self.global_r_ball, self.global_phi_ball = ball_obs
        self.finite_state = "TURN_TO_OFFSET"

        return {"left_motor": 0.0, "right_motor": 0.0}


    def turn_to_offset(self, sensors):
        ball, loa_unit, offset_point, loa_angle = self.get_loa_geometry(sensors)

        x_r, y_r, theta_r = self.mu
        dx = offset_point[0] - x_r
        dy = offset_point[1] - y_r

        target_angle = np.arctan2(dy, dx)
        heading_error = self.wrap(target_angle - theta_r)

        if abs(heading_error) < ANGLE_TOL:
            self.drive_line_frozen = False
            self.initial_side_error = None
            self.finite_state = "DRIVE_TO_OFFSET"
            return {"left_motor": 0.0, "right_motor": 0.0}

        direction = 1.0 if heading_error > 0 else -1.0

        return {
            "left_motor": -direction * TURN_SPEED,
            "right_motor": direction * TURN_SPEED
        }

    def drive_to_offset(self, sensors):
        x_r, y_r, theta_r = self.mu
        robot = np.array([x_r, y_r])

        live_ball, live_loa_unit, live_offset_point, live_loa_angle = self.get_loa_geometry(sensors)
        dist_to_ball = np.linalg.norm(live_ball - robot)

        if not self.drive_line_frozen and dist_to_ball < FREEZE_DISTANCE:
            self.drive_line_frozen = True
            self.drive_line_ball = live_ball
            self.drive_line_unit = live_loa_unit
            self.drive_offset_point = live_offset_point
            self.drive_line_angle = live_loa_angle

            robot_vec = robot - self.drive_line_ball
            self.initial_side_error = (
                self.drive_line_unit[0] * robot_vec[1]
                - self.drive_line_unit[1] * robot_vec[0]
            )

        if self.drive_line_frozen:
            ball = self.drive_line_ball
            loa_unit = self.drive_line_unit
            offset_point = self.drive_offset_point

            robot_vec = robot - ball
            current_side_error = (
                loa_unit[0] * robot_vec[1]
                - loa_unit[1] * robot_vec[0]
            )

            crossed_line = (
                self.initial_side_error is not None
                and current_side_error * self.initial_side_error <= 0
            )

            dist_to_offset = np.linalg.norm(offset_point - robot)

            if crossed_line or dist_to_offset < STAGING_TOL:
                self.drive_line_frozen = False
                self.initial_side_error = None
                self.finite_state = "TURN_TO_LOA"
                return {"left_motor": 0.0, "right_motor": 0.0}

        return {
            "left_motor": STAGE_SPEED,
            "right_motor": STAGE_SPEED
    }

    def turn_to_loa(self, sensors):
        if self.drive_line_angle is not None:
            loa_angle = self.drive_line_angle
        else:
            _, _, _, loa_angle = self.get_loa_geometry(sensors)

        theta_r = self.mu[2]
        heading_error = self.wrap(loa_angle - theta_r)

        if abs(heading_error) < ANGLE_TOL:
            self.finite_state = "PUSH_BALL"
            return {"left_motor": 0.0, "right_motor": 0.0}

        direction = 1.0 if heading_error > 0 else -1.0

        return {
            "left_motor": -direction * TURN_SPEED,
            "right_motor": direction * TURN_SPEED
        }

    def push_ball(self, sensors):
        _, _, _, loa_angle = self.get_loa_geometry(sensors)

        theta_r = self.mu[2]
        heading_error = self.wrap(loa_angle - theta_r)

        # tiny correction only
        turn = 1.0 * heading_error
        turn = max(-0.5, min(0.5, turn))

        left = PUSH_SPEED - turn
        right = PUSH_SPEED + turn

        ball_obs = sensors.get("ball", None)
        if ball_obs is None:
            self.finite_state = "BACK_UP"

        return {
            "left_motor": max(-MAX_SPEED, min(MAX_SPEED, left)),
            "right_motor": max(-MAX_SPEED, min(MAX_SPEED, right))
        }
    
    def back_up(self, sensors):
        ball_obs = sensors.get("ball", None)
        if ball_obs is not None:
            self.finite_state = "FIND_BALL"
        return {"left_motor": -MAX_SPEED, "right_motor": -MAX_SPEED}

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

            case "TURN_TO_OFFSET":
                control_dict = self.turn_to_offset(sensors)

            case "DRIVE_TO_OFFSET":
                control_dict = self.drive_to_offset(sensors)

            case "TURN_TO_LOA":
                control_dict = self.turn_to_loa(sensors)

            case "PUSH_BALL":
                control_dict = self.push_ball(sensors)

            case "BACK_UP":
                control_dict = self.back_up(sensors)

            case "DONE":
                control_dict = {"left_motor": 0.0, "right_motor": 0.0}

        # control_dict = self.pid_controller(sensors)

        # if (1.5 < self.mu[2] < 1.6):
        #     control_dict["left_motor"] = 6.5
        #     control_dict["right_motor"] = 6.5

        return control_dict