import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from carlike_control.wheel import Wheel
from typing import List
import math

# Define the transformation matrix
from carlike_control.helper import transform_2d


class Car:
    def __init__(self, x=0, y=0, yaw=0.0, v=0, length=0.56, width=0.56):
        ## shape of the car
        self.length = length
        self.width = width
        self.l_f = 0.28
        self.l_r = 0.28

        ## state of the car
        self.NX: int = 4
        self.x: float = x
        self.y: float = y
        self.yaw: float = yaw
        self.v: float = v

        ## command of the car
        self.NU: int = 3
        self.steer_front: float = 0
        self.steer_rear: float = 0
        self.accel: float = 0

        ## limit of the car
        self.MAX_STEER = np.deg2rad(45.0)  # maximum steering angle [rad]
        self.MAX_DSTEER = np.deg2rad(30.0)  # maximum steering speed [rad/s]
        self.MAX_SPEED = 0.5  # maximum speed [m/s]
        self.MIN_SPEED = -0.5  # minimum speed [m/s]
        self.MAX_ACCEL = 0.1  # maximum accel [m/ss]
        # for simulation
        self.body_points = []
        self.wheels: List[
            Wheel
        ] = []  ## List of wheels, left to right, front to back 0,1,2,3
        for j in [1, -1]:
            for i in [1, -1]:
                self.wheels.append(Wheel(x=j * self.length / 2, y=i * self.width / 2))
                self.body_points.append([j * self.length / 2, i * self.width / 2, 1])
        self.update_all_steer([0, 0, 0, 0])  # set all wheels to 0 steer
        self.body_points = np.array(self.body_points)

    def update_all_steer(self, theta_list):
        for i in range(len(self.wheels)):
            self.wheels[i].update_steer(theta_list[i])

    def update_all_steer_simple(self, front_steer):
        sign = np.sign(front_steer)
        beta = np.abs(front_steer)
        L = (self.l_f + self.l_r) / 2
        x = L / np.tan(beta)
        large_angle = np.arctan2(L, x - self.width / 2)
        small_angle = np.arctan2(L, x + self.width / 2)
        # print(f"large_angle: {large_angle}, small_angle: {small_angle}")
        if sign == -1:
            self.update_all_steer(
                [-small_angle, -large_angle, small_angle, large_angle]
            )
            return [-small_angle, -large_angle, small_angle, large_angle]
        else:
            self.update_all_steer(
                [large_angle, small_angle, -large_angle, -small_angle]
            )
            return [large_angle, small_angle, -large_angle, -small_angle]

    def update_pose(self, x, y, theta):
        self.x, self.y, self.yaw = x, y, theta

    def update_state(self, a, steer_front, steer_rear, DT=0.1):
        self.accel = a

        steer_rear = np.clip(steer_rear, -self.MAX_STEER, self.MAX_STEER)
        steer_front = np.clip(steer_front, -self.MAX_STEER, self.MAX_STEER)
        self.steer_front = steer_front
        self.steer_rear = steer_rear
        beta = self.calc_beta(steer_front, steer_rear)
        self.x = self.x + self.v * math.cos(self.yaw + beta) * DT
        self.y = self.y + self.v * math.sin(self.yaw + beta) * DT
        self.yaw = (
            self.yaw
            + self.v
            * np.cos(beta)
            * (np.tan(steer_front) - np.tan(steer_rear))
            / (self.l_r + self.l_f)
            * DT
        )
        self.v = self.v + a * DT

        self.v = np.clip(self.v, self.MIN_SPEED, self.MAX_SPEED)

        # return self

    def calc_beta(self, steer_front, steer_rear):
        return math.atan2(
            (self.l_f * np.tan(steer_rear) + self.l_r * np.tan(steer_front)),
            (self.l_f + self.l_r),
        )


if __name__ == "__main__":
    car = Car(x=5, y=5, yaw=0, length=4, width=2)
    print(f"body points in car frame: \n{car.body_points}")

    world_points = np.dot(car.w_v_transform, car.body_points.T).T
    print(f"body points in world frame: \n{world_points}")
