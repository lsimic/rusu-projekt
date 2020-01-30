from collections import namedtuple
import math
import numpy
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step

from utils import euclid_dist_2D, polar_to_cartesian

class EnvironmentV4(py_environment.PyEnvironment):
    def __init__(self):
        # all values are in meters, seconds, m/s or radians
        # values defining the car
        self.car_wheel_base = 4 
        self.car_max_speed = 40 
        self.car_max_steering_angle = math.radians(20)
        self.car_acceleration_force = 3000 # in N, force applied by the wheels
        self.car_braking_force = 9000 # in N, force applied by the wheels
        self.car_mass = 400 # in kg
        # values defining environment state
        self.car_location = (0.0, 0.0)
        self.car_heading = 0.0 # -pi to +pi, 0 is on +y
        self.car_speed = 0.0
        self.current_target_idx = 0
        self.next_target_idx = 1
        self.targets = []
        self.prev_location = self.car_location
        # values defining the simulation
        self.fps = 5
        self.time_step = 1/ self.fps
        self.playing_area_width = 100 # 100 meters
        self.playing_area_height = 100 # 100 meters
        self.target_count = 10
        self.visited_distance = 5 # distance whitin a target is considered visited
        self.steps_per_target = 14 * self.fps # 14 ~ diagonal of playing area / quarter of max speed
        self.max_steps = self.target_count * self.steps_per_target
        self.current_step = 0
        self.prev_step = 0
        # space specifications
        self._action_spec = array_spec.BoundedArraySpec(shape=(2,), dtype=numpy.float32, minimum=-1, maximum=1, name="action")
        self._observation_spec = array_spec.BoundedArraySpec(shape=(8,), dtype=numpy.float32, minimum=-1, maximum=1, name="observation")
        self._reward_spec = array_spec.BoundedArraySpec(shape=(1,), dtype=numpy.float32, minimum=-5, maximum=20.0, name="reward")
        self._discount_spec = array_spec.BoundedArraySpec(shape=(1,), dtype=numpy.float32, minimum=0.0, maximum=1.0, name="discount")
        self._time_step_spec = namedtuple("time_step_spec", ["action", "observation", "reward", "discount"])
        self._time_step_spec.action = self._action_spec
        self._time_step_spec.observation = self._observation_spec
        self._time_step_spec.reward = self._reward_spec
        self._time_step_spec.discount = self._discount_spec
        # reset the environment on initialization
        self._reset()

    def observation_spec(self):
        # observation = [cuurrent_target_x, current_target_y, next_target_x, next_target_y, pos_x, pos_y, velocity_x, velocity_y]
        return self._observation_spec

    def action_spec(self):
        # action = [throttle_or_brake, turn_left_or_right]
        return self._action_spec

    def _reset(self):
        self.prev_step = 0
        self.current_step = 0
        self.current_target_idx = 0
        self.next_target_idx = 1
        self.targets = []
        for i in range(0, self.target_count):
            self.targets.append((
                numpy.random.uniform(-0.3 * self.playing_area_width, 0.3 * self.playing_area_width), 
                numpy.random.uniform(-0.3 * self.playing_area_height, 0.3 * self.playing_area_height)))
        self.targets.append(self.targets[-1])
        self.car_location = (
            numpy.random.uniform(-0.3 * self.playing_area_width, 0.3 * self.playing_area_width), 
            numpy.random.uniform(-0.3 * self.playing_area_height, 0.3 * self.playing_area_height))
        self.car_speed = numpy.random.uniform(0.25 * self.car_max_speed, 0.75 * self.car_max_speed)
        self.car_heading = numpy.random.uniform(-math.pi, math.pi)
        self.prev_location = self.car_location
        # create observation, return
        vel_cart = polar_to_cartesian((self.car_speed, self.car_heading))
        observation = numpy.array([
            self.targets[self.current_target_idx][0] / (0.5 * self.playing_area_width), self.targets[self.current_target_idx][1] / (0.5 * self.playing_area_height),
            self.targets[self.next_target_idx][0] / (0.5 * self.playing_area_width), self.targets[self.next_target_idx][1] / (0.5 * self.playing_area_height),
            self.car_location[0] / (0.5 * self.playing_area_width), self.car_location[1] / (0.5 * self.playing_area_height),
            vel_cart[0] / self.car_max_speed, vel_cart[1] / self.car_max_speed,
        ])
        observation = observation.astype(numpy.float32)
        return time_step.restart(observation)

    def _step(self, action):
        # calculate initial distance to target position
        distance_initial = euclid_dist_2D(self.targets[self.current_target_idx], self.car_location)
        # apply acceleration, calculate new speed and current steering angle.
        # increase steering angle if going slower
        steering_agnle = self.car_max_steering_angle * action[1]
        if self.car_speed > 0:
            steering_agnle *= (self.car_max_speed / self.car_speed)
        if action[0] > 0:
            acceleration = (self.car_acceleration_force * action[0]) / self.car_mass
        else:
            acceleration = (self.car_braking_force * action[0]) / self.car_mass
        self.car_speed = self.car_speed + acceleration * self.time_step
        if self.car_speed > self.car_max_speed:
            self.car_speed = self.car_max_speed
        if self.car_speed < 0:
            self.car_speed = 0
        # calculate location of front and rear axles.
        front_axle = (
            self.car_location[0] + (self.car_wheel_base / 2) * math.cos(self.car_heading), 
            self.car_location[1] + (self.car_wheel_base / 2) * math.sin(self.car_heading))
        rear_axle = (
            self.car_location[0] - (self.car_wheel_base / 2) * math.cos(self.car_heading), 
            self.car_location[1] - (self.car_wheel_base / 2) * math.sin(self.car_heading))
        # calculate new location of front and rear axles
        loc_rear = (
            rear_axle[0] + self.car_speed * self.time_step * math.cos(self.car_heading),
            rear_axle[1] + self.car_speed * self.time_step * math.sin(self.car_heading))
        loc_front = (
            front_axle[0] + self.car_speed * self.time_step * math.cos(self.car_heading + steering_agnle),
            front_axle[1] + self.car_speed * self.time_step * math.sin(self.car_heading + steering_agnle))
        # calculate new location and heading, update environment state values
        self.car_location = (
            (loc_front[0] + loc_rear[0]) / 2,
            (loc_front[1] + loc_rear[1]) / 2)
        self.car_heading = math.atan2(loc_front[1] - loc_rear[1], loc_front[0] - loc_rear[0])

        # calculate current distance to target position
        distance_current = euclid_dist_2D(self.targets[self.current_target_idx], self.car_location)

        # calculate reward, check completion criteria
        max_possible_dist = self.car_max_speed * self.time_step
        reward = 0
        done = False
        if distance_current < distance_initial:
            reward = 0.1 * (distance_current-distance_initial) / max_possible_dist
        else:
            reward = -0.1
        # check if car is within bounds
        if abs(self.car_location[0]) > 0.5 * self.playing_area_width or abs(self.car_location[1]) > 0.5 * self.playing_area_height:
            done = True
            reward = -5
        # check if car has reached the current target
        if distance_current < self.visited_distance:
            reward = 5 + 5 * max(0, 1 - (self.current_step - self.prev_step) / self.steps_per_target)
            self.prev_step = self.current_step
            self.prev_location = self.targets[self.current_target_idx]
            if self.current_target_idx == self.target_count - 1:
                done = True
                reward += 10 * (self.max_steps - self.current_step) / self.max_steps
            else:
                self.current_target_idx += 1
                self.next_target_idx += 1
        # check if it's taking too long to reach a target
        if self.current_step - self.prev_step > 3*self.steps_per_target:
            done=True
            reward = -5
        # check if max steps were reached
        if self.current_step >= self.max_steps:
            done = True
            dist_start = euclid_dist_2D(self.targets[self.current_target_idx], self.prev_location)
            if dist_start:
                reward = 5 * (1 - distance_current / dist_start)
            else:
                reward += 10 * (self.max_steps - self.current_step) / self.max_steps
            if reward < -5:
                reward = -5

        # create observation, return...
        vel_cart = polar_to_cartesian((self.car_speed, self.car_heading))
        observation = numpy.array([
            self.targets[self.current_target_idx][0] / (0.5 * self.playing_area_width), self.targets[self.current_target_idx][1] / (0.5 * self.playing_area_height),
            self.targets[self.next_target_idx][0] / (0.5 * self.playing_area_width), self.targets[self.next_target_idx][1] / (0.5 * self.playing_area_height),
            self.car_location[0] / (0.5 * self.playing_area_width), self.car_location[1] / (0.5 * self.playing_area_height),
            vel_cart[0] / self.car_max_speed, vel_cart[1] / self.car_max_speed,
        ])
        observation = observation.astype(numpy.float32)
        # increment step, return 
        self.current_step += 1
        if done:
            # print("completed at step " + str(self.current_step))
            return time_step.termination(observation, reward)
        else:
            return time_step.transition(observation, reward, discount=0.98)

    def seed(self, seed=None):
        numpy.random.seed(seed)
