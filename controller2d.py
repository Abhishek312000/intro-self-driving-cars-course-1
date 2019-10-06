#!/usr/bin/env python3

"""
2D Controller Class to be used for the CARLA waypoint follower demo.
"""

import cutils
import numpy as np

class Controller2D(object):
    def __init__(self, waypoints):
        self.vars                  = cutils.CUtils()
        self._current_x            = 0
        self._current_y            = 0
        self._current_yaw          = 0
        self._current_speed        = 0
        self._desired_speed        = 0
        self._current_frame        = 0
        self._current_timestamp    = 0
        self._start_control_loop   = False
        self._set_throttle         = 0
        self._set_brake            = 0
        self._set_steer            = 0
        self._waypoints            = waypoints
        self._conv_rad_to_steer    = 180.0 / 70.0 / np.pi
        self._pi                   = np.pi
        self._2pi                  = 2.0 * np.pi
        self._target_waypoint_idx  = 0
        self._target_waypoint_dist = 0

    def update_values(self, x, y, yaw, speed, timestamp, frame):
        self._current_x         = x
        self._current_y         = y
        self._current_yaw       = yaw
        self._current_speed     = speed
        self._current_timestamp = timestamp
        self._current_frame     = frame
        if self._current_frame:
            self._start_control_loop = True

    def update_desired_speed(self):
        min_idx       = 0
        min_dist      = float("inf")
        desired_speed = 0
        for i in range(len(self._waypoints)):
            dist = np.linalg.norm(np.array([
                    self._waypoints[i][0] - self._current_x,
                    self._waypoints[i][1] - self._current_y]))
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        if min_idx < len(self._waypoints)-1:
            desired_speed = self._waypoints[min_idx][2]
            self._target_waypoint_idx = min_idx
            self._target_waypoint_dist = min_dist
        else:
            desired_speed = self._waypoints[-1][2]
            self._target_waypoint_idx = -1
            self._target_waypoint_dist = np.linalg.norm(np.array([
                    self._waypoints[-1][0] - self._current_x,
                    self._waypoints[-1][1] - self._current_y]))

        self._desired_speed = desired_speed
        

    def update_waypoints(self, new_waypoints):
        self._waypoints = new_waypoints

    def get_commands(self):
        return self._set_throttle, self._set_steer, self._set_brake

    def set_throttle(self, input_throttle):
        # Clamp the throttle command to valid bounds
        throttle           = np.fmax(np.fmin(input_throttle, 1.0), 0.0)
        self._set_throttle = throttle

    def set_steer(self, input_steer_in_rad):
        # Covnert radians to [-1, 1]
        input_steer = self._conv_rad_to_steer * input_steer_in_rad

        # Clamp the steering command to valid bounds
        steer           = np.fmax(np.fmin(input_steer, 1.0), -1.0)
        self._set_steer = steer

    def set_brake(self, input_brake):
        # Clamp the steering command to valid bounds
        brake           = np.fmax(np.fmin(input_brake, 1.0), 0.0)
        self._set_brake = brake

    def update_controls(self):
        ######################################################
        # RETRIEVE SIMULATOR FEEDBACK
        ######################################################
        x               = self._current_x
        y               = self._current_y
        yaw             = self._current_yaw
        v               = self._current_speed
        self.update_desired_speed()
        v_desired       = self._desired_speed
        t               = self._current_timestamp
        waypoints       = self._waypoints
        throttle_output = 0
        steer_output    = 0
        brake_output    = 0
      
        ######################################################
        ######################################################
        # MODULE 7: DECLARE USAGE VARIABLES HERE
        ######################################################
        ######################################################
        """
            Use 'self.vars.create_var(<variable name>, <default value>)'
            to create a persistent variable (not destroyed at each iteration).
            This means that the value can be stored for use in the next
            iteration of the control loop.

            Example: Creation of 'v_previous', default value to be 0
            self.vars.create_var('v_previous', 0.0)

            Example: Setting 'v_previous' to be 1.0
            self.vars.v_previous = 1.0

            Example: Accessing the value from 'v_previous' to be used
            throttle_output = 0.5 * self.vars.v_previous
        """
        self.vars.create_var('t_previous', 0.0)
        self.vars.create_var('steady_state_yaw', 0.0)
        
        self.vars.create_var('v_last_error', 0.0)
        self.vars.create_var('v_cumulative_error', 0.0)
      
        # Skip the first frame to store previous values properly
        if self._start_control_loop:
            """
                Controller iteration code block.

                Controller Feedback Variables:
                    x               : Current X position (meters)
                    y               : Current Y position (meters)
                    yaw             : Current yaw pose (radians)
                    v               : Current forward speed (meters per second)
                    t               : Current time (seconds)
                    v_desired       : Current desired speed (meters per second)
                                      (Computed as the speed to track at the
                                      closest waypoint to the vehicle.)
                    waypoints       : Current waypoints to track
                                      (Includes speed to track at each x,y
                                      location.)
                                      Format: [[x0, y0, v0],
                                               [x1, y1, v1],
                                               ...
                                               [xn, yn, vn]]
                                      Example:
                                          waypoints[2][1]: 
                                          Returns the 3rd waypoint's y position

                                          waypoints[5]:
                                          Returns [x5, y5, v5] (6th waypoint)
                
                Controller Output Variables:
                    throttle_output : Throttle output (0 to 1)
                    steer_output    : Steer output (-1.22 rad to 1.22 rad)
                    brake_output    : Brake output (0 to 1)
            """

            print(f"x={x}; y={y}; yaw={yaw}; v={v}; v_desired={v_desired}")
                
            ######################################################
            ######################################################
            # MODULE 7: IMPLEMENTATION OF LONGITUDINAL CONTROLLER HERE
            ######################################################
            ######################################################
            t_delta = t - self.vars.t_previous

            v_error = v_desired - v
            self.vars.v_cumulative_error += v_error

            Kp = 0.8
            Ki = 0.5
            Kd = 0.1
            P = Kp * v_error
            I = Ki * self.vars.v_cumulative_error * t_delta
            D = Kd * (v_error - self.vars.v_last_error) / t_delta
            a_desired = P + I + D
            if (a_desired > 0):
                throttle_output = a_desired
                brake_output = 0
            else:
                throttle_output = 0
                brake_output = a_desired

            self.vars.v_last_error = v_error            

            ######################################################
            ######################################################
            # MODULE 7: IMPLEMENTATION OF LATERAL CONTROLLER HERE
            ######################################################
            ######################################################
            if self._target_waypoint_idx != -1:
                previous_waypoint = self._waypoints[self._target_waypoint_idx - 1]
                target_waypoint = self._waypoints[len(self._waypoints) - 1]
                
                # mov position to front axis
                # L = 4.613 * 0.25
                # x += L * np.cos(yaw)
                # y += L * np.sin(yaw)

                # calculate crosstrack error
                previous_point = np.array(previous_waypoint[:2])
                target_point = np.array(target_waypoint[:2])
                current_point = np.array([x, y])

                print(f"    Points: prev={previous_point}; current={current_point}; target={target_point};")

                previos_to_current = current_point - previous_point
                previos_to_target = target_point - previous_point

                previos_to_current_dist = np.linalg.norm(previos_to_current)
                previos_to_targer_dist = np.linalg.norm(previos_to_target)

                vector_angle_cos = np.dot(previos_to_current, previos_to_target) / previos_to_current_dist / previos_to_targer_dist
                vector_angle_sin = np.sqrt(1 - vector_angle_cos * vector_angle_cos)

                crosstrack_error = vector_angle_sin * previos_to_current_dist
                
                # calculate heading error
                trajectory_angle = np.arctan((target_waypoint[1] - previous_waypoint[1]) / (target_waypoint[0] - previous_waypoint[0]))
                heading_error = yaw - trajectory_angle

                k_gain = 1
                k_soft = 1
                crosstrack_regulator = np.arctan(k_gain * crosstrack_error / (k_soft + v))
                steer_output = heading_error + crosstrack_regulator
                
                print(f" - error={crosstrack_error}; cross_regulator={crosstrack_regulator}; heading_error={heading_error}; heading={yaw} trajectory_angle={trajectory_angle}; steer={steer_output}")
            else:
                steer_output = 0

            
            ######################################################
            # SET CONTROLS OUTPUT
            ######################################################
            self.set_throttle(throttle_output)  # in percent (0 to 1)
            self.set_steer(steer_output)        # in rad (-1.22 to 1.22)
            self.set_brake(brake_output)        # in percent (0 to 1)
        else:
            self.vars.steady_state_yaw = yaw

        ######################################################
        ######################################################
        # MODULE 7: STORE OLD VALUES HERE (ADD MORE IF NECESSARY)
        ######################################################
        ######################################################
        self.vars.t_previous = t
        