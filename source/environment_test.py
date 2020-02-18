from collections import namedtuple
import math
import numpy
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step

from utils import euclid_dist_2D, polar_to_cartesian, normalized, angle_signed

class TestEnvironment(py_environment.PyEnvironment):
    def __init__(self):
        # all values are in meters, seconds, m/s or radians
        # values defining the car
        self.car_wheel_base = 4 
        self.car_speed = 20.0
        self.car_steering_angle = math.radians(20)
        # values defining environment state
        self.car_location = (0.0, 0.0)
        self.car_heading = 0.0 # -pi to +pi, 0 is on +y
        self.target = (0.0, 0.0)
        self.initial_location = self.car_location
        # values defining the simulation
        self.fps = 15
        self.time_step = 1 / self.fps
        self.playing_area_width = 100 # 100 meters
        self.playing_area_height = 100 # 100 meters
        self.visited_distance = 5.0 # distance whitin a target is considered visited
        self.max_steps = 150
        self.current_step = 0
        self.max_target_count = 5
        self.current_target = 0
        # space specifications
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=numpy.int32, minimum=0, maximum=2, name="action")
        self._observation_spec = array_spec.BoundedArraySpec(shape=(7,), dtype=numpy.float32, minimum=-1, maximum=1, name="observation")
        self._reward_spec = array_spec.BoundedArraySpec(shape=(1,), dtype=numpy.float32, minimum=-100, maximum=100.0, name="reward")
        self._discount_spec = array_spec.BoundedArraySpec(shape=(1,), dtype=numpy.float32, minimum=0.0, maximum=1.0, name="discount")
        self._time_step_spec = namedtuple("time_step_spec", ["action", "observation", "reward", "discount"])
        self._time_step_spec.action = self._action_spec
        self._time_step_spec.observation = self._observation_spec
        self._time_step_spec.reward = self._reward_spec
        self._time_step_spec.discount = self._discount_spec
        # reset the environment on initialization
        self._reset()

    def observation_spec(self):
        # observation = [target_x, target_y, pos_x, pos_y, velocity_x, velocity_y]
        return self._observation_spec

    def action_spec(self):
        # action = [throttle_or_brake, turn_left_or_right]
        return self._action_spec

    def _reset(self):
        self.current_target = 0
        self.current_step = 0
        self.target = (
            numpy.random.uniform(-0.25 * self.playing_area_width, 0.25 * self.playing_area_width), 
            numpy.random.uniform(-0.25 * self.playing_area_height, 0.25 * self.playing_area_height)
        )
        self.car_location = (
            numpy.random.uniform(-0.25 * self.playing_area_width, 0.25 * self.playing_area_width), 
            numpy.random.uniform(-0.25 * self.playing_area_height, 0.25 * self.playing_area_height)
        )
        self.car_heading = numpy.random.uniform(-math.pi, math.pi)
        self.initial_location = self.car_location

        desired_vector = (self.target[0] - self.car_location[0], self.target[1] - self.car_location[1])
        desired_vector = normalized(desired_vector)
        current_vector = polar_to_cartesian((1, self.car_heading))
        angle_difference_end = angle_signed(desired_vector, current_vector)

        # create observation, return
        vel_cart = polar_to_cartesian((1, self.car_heading))
        observation = numpy.array([
            self.target[0] / (0.5 * self.playing_area_width), self.target[1] / (0.5 * self.playing_area_height),
            self.car_location[0] / (0.5 * self.playing_area_width), self.car_location[1] / (0.5 * self.playing_area_height),
            vel_cart[0], vel_cart[1],
            angle_difference_end / math.pi
        ])
        observation = observation.astype(numpy.float32)
        return time_step.restart(observation)

    def _step(self, action):
        # calculate difference between current heading and desired heading
        desired_vector = (self.target[0] - self.car_location[0], self.target[1] - self.car_location[1])
        desired_vector = normalized(desired_vector)
        current_vector = polar_to_cartesian((1, self.car_heading))
        angle_difference_start = angle_signed(desired_vector, current_vector)

        # calculate angle change
        angle = 0
        # if action == 0 continue straight...
        if action == 1: # turn left
            angle = - self.car_steering_angle
        if action == 2: # turn right
            angle = self.car_steering_angle

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
            front_axle[0] + self.car_speed * self.time_step * math.cos(self.car_heading + angle),
            front_axle[1] + self.car_speed * self.time_step * math.sin(self.car_heading + angle))
        # calculate new location and heading, update environment state values
        self.car_location = (
            (loc_front[0] + loc_rear[0]) / 2,
            (loc_front[1] + loc_rear[1]) / 2)
        self.car_heading = math.atan2(loc_front[1] - loc_rear[1], loc_front[0] - loc_rear[0])

        # calculate difference between current heading and desired heading
        desired_vector = (self.target[0] - self.car_location[0], self.target[1] - self.car_location[1])
        desired_vector = normalized(desired_vector)
        current_vector = polar_to_cartesian((1, self.car_heading))
        angle_difference_end = angle_signed(desired_vector, current_vector)

        done = False
        angle_change = abs(angle_difference_start) - abs(angle_difference_end)
        reward = angle_change / math.radians(self.car_steering_angle)

        # check if car is within bounds
        if abs(self.car_location[0]) > 0.5 * self.playing_area_width or abs(self.car_location[1]) > 0.5 * self.playing_area_height:
            done = True
            reward = -100
        # check if car has reached the current target
        distance = euclid_dist_2D(self.target, self.car_location)
        if distance < self.visited_distance:
            reward = 25 + 75 * (self.current_step / self.max_steps)
            # add new target and reset step...
            self.current_target += 1
            if self.current_target < self.max_target_count:
                self.target = (
                    numpy.random.uniform(-0.25 * self.playing_area_width, 0.25 * self.playing_area_width), 
                    numpy.random.uniform(-0.25 * self.playing_area_height, 0.25 * self.playing_area_height)
                )
                self.current_step = 0
            else:
                done = True
        # check if max steps were reached
        if self.current_step >= self.max_steps:
            done = True
            distance_initial = euclid_dist_2D(self.target, self.initial_location)
            if distance_initial:
                reward = 25 * (1 - distance / distance_initial)
            else:
                reward = 2.5

        # create observation, return...
        vel_cart = polar_to_cartesian((1, self.car_heading))
        observation = numpy.array([
            self.target[0] / (0.5 * self.playing_area_width), self.target[1] / (0.5 * self.playing_area_height),
            self.car_location[0] / (0.5 * self.playing_area_width), self.car_location[1] / (0.5 * self.playing_area_height),
            vel_cart[0], vel_cart[1],
            angle_difference_end / math.radians(180)
        ])
        observation = observation.astype(numpy.float32)
        # increment step, return 
        self.current_step += 1
        # print(reward)
        if done:
            return time_step.termination(observation, reward)
        else:
            return time_step.transition(observation, reward, discount=0.98)

    def seed(self, seed=None):
        numpy.random.seed(seed)
