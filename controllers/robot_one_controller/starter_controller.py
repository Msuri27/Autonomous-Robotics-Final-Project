"""student_controller controller."""

import math
import numpy as np

MAX_SPEED = 6.5

# Line-of-action staging
LOA_OFFSET = 0.3       # distance behind ball toward opposite side of goal
STAGING_TOL = 0.15      # how close robot must get to offset point
FREEZE_DISTANCE = 0.8
LINE_TOL = 0.08

# Turning thresholds
ANGLE_TOL = 0.06        # heading tolerance for turn-in-place states

# Goal/end condition
GOAL_REACHED_DIST = 0.25

# Speeds
TURN_SPEED = 4.0        # in-place turning speed
STAGE_SPEED = 5.0       # drive-to-offset speed
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

        self.initial_line_error = None
        self.frozen_line_vertical = False
        self.frozen_m = 0.0
        self.frozen_b = 0.0
        self.frozen_x = 0.0

        self.must_avoid = False

        self.avoid_mode = "TURN"
        self.avoid_start_y = None
        self.avoid_target_heading = None
        self.avoid_direction = 1.0
        self.avoid_checked = False

        self.startup_checked = False
        self.must_avoid = False
        self.avoid_heading = None
        self.avoid_target_y = None
        self.avoid_cushion = 0.5

        self.refind_used = False
        self.refind_margin = 0.15

        self.prev_stuck_pose = None
        self.stuck_counter = 0
        self.backup_start_pose = None

        self.stuck_pos_tol = 0.01      # meters per step
        self.stuck_theta_tol = 0.02    # radians per step
        self.stuck_steps_needed = 30   # about 0.5 sec at 32 ms

        self.backup_distance = 0.25

        self.steps_since_observation = 0
        self.observation_timeout_steps = int(20.0 / 0.032)   # ~30 seconds at 32 ms timestep

        self.pose_history = []
        self.stuck_window_steps = int(10.0 / 0.032)   # 10 seconds
        self.stuck_net_disp_tol = 0.25                # meters over window
        self.steps_since_observation = 0
        self.observation_timeout_steps = int(10.0 / 0.032)

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
        x_r, y_r, _ = self.mu

        if ball_obs is None:
            return {"left_motor": -MAX_SPEED, "right_motor": MAX_SPEED}

        self.global_r_ball, self.global_phi_ball = ball_obs

        # ONLY do this avoidance decision once at startup
        if not self.startup_checked:
            ball_coords = self.get_ball_coords(sensors)
            ball_x, ball_y = ball_coords

            ball_behind_robot = ball_x < x_r
            ball_behind_goal = ball_x > self.goal_coords[0]

            if ball_behind_robot or ball_behind_goal:
                self.must_avoid = True

                if ball_y >= y_r:
                    self.avoid_heading = np.pi / 2      # 90 deg, go up
                    self.avoid_target_y = ball_y + self.avoid_cushion
                else:
                    self.avoid_heading = -np.pi / 2     # 270 deg, go down
                    self.avoid_target_y = ball_y - self.avoid_cushion

                self.finite_state = "DRIVE_TO_POINT"
            else:
                self.must_avoid = False
                self.finite_state = "TURN_TO_OFFSET"

            self.startup_checked = True
            return {"left_motor": 0.0, "right_motor": 0.0}

        # after startup check, just continue normal behavior
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

    def compute_drive_point(self, sensors):
        x_r, y_r, theta_r = self.mu

        if self.avoid_heading is None or self.avoid_target_y is None:
            self.must_avoid = False
            self.finite_state = "FIND_BALL"
            return {"left_motor": 0.0, "right_motor": 0.0}

        heading_error = self.wrap(self.avoid_heading - theta_r)

        # Step 1: turn in place to vertical
        if abs(heading_error) > 0.12:
            if heading_error > 0:
                return {
                    "left_motor": -TURN_SPEED,
                    "right_motor": TURN_SPEED
                }
            else:
                return {
                    "left_motor": TURN_SPEED,
                    "right_motor": -TURN_SPEED
                }

        # Step 2: drive straight vertically until target y is reached
        if self.avoid_heading > 0:  # going up
            reached_target = y_r >= self.avoid_target_y
        else:                       # going down
            reached_target = y_r <= self.avoid_target_y

        if reached_target:
            self.must_avoid = False
            self.avoid_heading = None
            self.avoid_target_y = None
            self.finite_state = "FIND_BALL"
            return {"left_motor": 0.0, "right_motor": 0.0}

        return {
            "left_motor": STAGE_SPEED,
            "right_motor": STAGE_SPEED
        }

    def drive_to_point(self, target_point):
        x_r, y_r, theta_r = self.mu
        target = np.array(target_point)

        dx = target[0] - x_r
        dy = target[1] - y_r
        dist = np.hypot(dx, dy)

        target_angle = np.arctan2(dy, dx)
        heading_error = self.wrap(target_angle - theta_r)

        # turn in place first
        if abs(heading_error) > ANGLE_TOL:
            direction = 1.0 if heading_error > 0 else -1.0
            return {
                "left_motor": -direction * TURN_SPEED,
                "right_motor": direction * TURN_SPEED
            }, False

        # close enough
        if dist < STAGING_TOL:
            return {
                "left_motor": 0.0,
                "right_motor": 0.0
            }, True

        # drive straight once aligned
        return {
            "left_motor": STAGE_SPEED,
            "right_motor": STAGE_SPEED
        }, False

    def drive_to_offset(self, sensors):
        x_r, y_r, theta_r = self.mu
        robot = np.array([x_r, y_r])

        live_ball, live_loa_unit, live_offset_point, live_loa_angle = self.get_loa_geometry(sensors)
        dist_to_ball = np.linalg.norm(live_ball - robot)

        ball_x = live_ball[0]

        if (not self.refind_used) and (x_r < ball_x - self.refind_margin):
            self.refind_used = True
            self.drive_line_frozen = False
            self.initial_line_error = None
            self.finite_state = "FIND_BALL"
            return {"left_motor": 0.0, "right_motor": 0.0}

        if self.must_avoid:
            self.finite_state = "DRIVE_TO_POINT"
            return {"left_motor": 0.0, "right_motor": 0.0}
        
        if not self.drive_line_frozen and dist_to_ball < FREEZE_DISTANCE:
            self.drive_line_frozen = True
            self.drive_line_ball = live_ball
            self.drive_line_unit = live_loa_unit
            self.drive_offset_point = live_offset_point
            self.drive_line_angle = live_loa_angle

            # Build frozen line equation
            x_b, y_b = live_ball
            dx, dy = live_loa_unit

            if abs(dx) > 1e-6:
                self.frozen_line_vertical = False
                self.frozen_m = dy / dx
                self.frozen_b = y_b - self.frozen_m * x_b

                y_line = self.frozen_m * x_r + self.frozen_b
                self.initial_line_error = y_r - y_line
            else:
                self.frozen_line_vertical = True
                self.frozen_x = x_b
                self.initial_line_error = x_r - self.frozen_x

            print(f"FROZEN_OFFSET_POINT: {live_offset_point}")

        if self.drive_line_frozen:
            offset_point = self.drive_offset_point

            if self.frozen_line_vertical:
                current_line_error = x_r - self.frozen_x
                line_dist = abs(current_line_error)
            else:
                y_line = self.frozen_m * x_r + self.frozen_b
                current_line_error = y_r - y_line
                line_dist = abs(current_line_error)

            crossed_line = (
                self.initial_line_error is not None
                and current_line_error * self.initial_line_error <= 0
            )

            dist_to_offset = np.linalg.norm(offset_point - robot)

            print(
                "LINE_CHECK:",
                "current_error", current_line_error,
                "initial_error", self.initial_line_error,
                "crossed", crossed_line,
                "line_dist", line_dist,
                "dist_offset", dist_to_offset
            )

            if crossed_line or line_dist < LINE_TOL or dist_to_offset < STAGING_TOL:
                self.drive_line_frozen = False
                self.initial_line_error = None
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
        ball_coords = self.get_ball_coords(sensors)
        if ball_coords[0] > (self.goal_coords[0] + 0.1) and -0.7 < ball_coords[1] < 0.7:
            self.finite_state = "DONE"

        return {
            "left_motor": max(-MAX_SPEED, min(MAX_SPEED, left)),
            "right_motor": max(-MAX_SPEED, min(MAX_SPEED, right))
        }
    
    def back_up(self, sensors):
        ball_obs = sensors.get("ball", None)
        if ball_obs is not None:
            self.finite_state = "FIND_BALL"
        return {"left_motor": -MAX_SPEED, "right_motor": -MAX_SPEED}

    def check_stuck(self):
        x_r, y_r, theta_r = self.mu

        if self.prev_stuck_pose is None:
            self.prev_stuck_pose = np.array([x_r, y_r, theta_r])
            return False

        dx = x_r - self.prev_stuck_pose[0]
        dy = y_r - self.prev_stuck_pose[1]
        dpos = np.hypot(dx, dy)
        dtheta = abs(self.wrap(theta_r - self.prev_stuck_pose[2]))

        self.prev_stuck_pose = np.array([x_r, y_r, theta_r])

        if dpos < self.stuck_pos_tol and dtheta < self.stuck_theta_tol:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        return self.stuck_counter >= self.stuck_steps_needed
    
    def back_up_recover(self, sensors):
        x_r, y_r, _ = self.mu

        if self.backup_start_pose is None:
            self.backup_start_pose = np.array([x_r, y_r])

        dist = np.linalg.norm(np.array([x_r, y_r]) - self.backup_start_pose)

        if dist >= self.backup_distance:
            self.backup_start_pose = None
            self.stuck_counter = 0
            self.prev_stuck_pose = None
            self.finite_state = "FIND_BALL"
            return {"left_motor": 0.0, "right_motor": 0.0}

        return {
            "left_motor": -STAGE_SPEED,
            "right_motor": -STAGE_SPEED
        }
    
    def update_observation_timer(self, sensors):
        saw_anything = False

        # ball counts
        if sensors.get("ball", None) is not None:
            saw_anything = True

        # landmarks count
        for key in ["center_circle", "goal", "penalty_cross", "corners"]:
            if key in sensors:
                try:
                    val = sensors[key]
                    if val is not None:
                        if key == "center_circle":
                            if len(val) == 2:
                                saw_anything = True
                        else:
                            if len(val) > 0:
                                saw_anything = True
                except:
                    pass

        if saw_anything:
            self.steps_since_observation = 0
        else:
            self.steps_since_observation += 1

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
        self.update_observation_timer(sensors)

        estimated_pose = self.mu.tolist()
        print(estimated_pose)
        print(self.finite_state)

        if self.finite_state in ["TURN_TO_OFFSET", "DRIVE_TO_OFFSET", "TURN_TO_LOA", "PUSH_BALL"]:
            if self.check_stuck():
                self.finite_state = "BACK_UP_RECOVER"
                self.backup_start_pose = None
        
        if self.steps_since_observation > self.observation_timeout_steps:
            self.finite_state = "BACK_UP_RECOVER"
            self.backup_start_pose = None
            self.steps_since_observation = 0

        match self.finite_state:
            case "FIND_BALL":
                control_dict = self.find_ball(sensors)

            case "TURN_TO_OFFSET":
                control_dict = self.turn_to_offset(sensors)

            case "DRIVE_TO_OFFSET":
                control_dict = self.drive_to_offset(sensors)

            case "DRIVE_TO_POINT":
                control_dict = self.compute_drive_point(sensors)

            case "TURN_TO_LOA":
                control_dict = self.turn_to_loa(sensors)

            case "PUSH_BALL":
                control_dict = self.push_ball(sensors)

            case "BACK_UP":
                control_dict = self.back_up(sensors)

            case "BACK_UP_RECOVER":
                control_dict = self.back_up_recover(sensors)

            case "DONE":
                control_dict = {"left_motor": 0.0, "right_motor": 0.0}

        # control_dict = self.pid_controller(sensors)

        # if (1.5 < self.mu[2] < 1.6):
        #     control_dict["left_motor"] = 6.5
        #     control_dict["right_motor"] = 6.5

        return control_dict