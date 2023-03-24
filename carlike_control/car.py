import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from carlike_control.wheel import Wheel
from typing import List
import math

# Define the transformation matrix
from carlike_control.helper import transform_2d


class Car:
    def __init__(self, x=0, y=0, yaw=0.0, v=0, length=4, width=2):
        ## shape of the car
        self.length = length
        self.width = width

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

    def update_pose(self, x, y, theta):
        self.x, self.y, self.yaw = x, y, theta

    @staticmethod
    def calc_beta(steer_front, steer_rear, l_f, l_r):
        return math.atan2(
            (l_f * np.tan(steer_rear) + l_r * np.tan(steer_front)), (l_f + l_r)
        )

    @staticmethod
    def get_linear_model_matrix(
        v: float,
        yaw: float,
        steer_front: float,
        steer_rear: float,
        DT=0.1,
    ):
        beta = Car.calc_beta(steer_front, steer_rear)
        A = np.identity(4)
        A[0, 2] = DT * math.cos(yaw + beta)
        A[0, 3] = -DT * v * math.sin(yaw + beta)
        A[1, 2] = DT * math.sin(yaw + beta)
        A[1, 3] = DT * v * math.cos(yaw + beta)
        A[3, 2] = (
            DT * np.cos(beta) * (np.tan(steer_front) - np.tan(steer_rear)) / (l_r + l_f)
        )

        B = np.zeros((4, 2))
        B[2, 0] = DT
        B[3, 1] = DT * v * np.cos(beta) / (l_r + l_f) / np.cos(steer_front) ** 2
        B[3, 2] = -DT * v * np.cos(beta) / (l_r + l_f) / np.cos(steer_rear) ** 2

        C = np.zeros(4)
        C[0] = DT * v * math.sin(yaw + beta) * yaw
        C[1] = -DT * v * math.cos(yaw + beta) * yaw
        C[3] = (
            -DT
            * v
            * np.cos(beta)
            / (l_r + l_f)
            * (1 / np.cos(steer_front) ** 2 - 1 / np.cos(steer_rear) ** 2)
        )
        return A, B, C


if __name__ == "__main__":

    car = Car(x=5, y=5, yaw=0, length=4, width=2)
    print(f"body points in car frame: \n{car.body_points}")

    world_points = np.dot(car.w_v_transform, car.body_points.T).T
    print(f"body points in world frame: \n{world_points}")
