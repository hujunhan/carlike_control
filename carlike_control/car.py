import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from carlike_control.wheel import Wheel
from typing import List

# Define the transformation matrix
from carlike_control.helper import transform_2d


class Car:
    def __init__(self, x=0, y=0, theta=0.0, length=4, width=2):
        self.x = x
        self.y = y
        self.theta = theta
        self.length = length
        self.width = width
        self.w_v_transform = transform_2d(x, y, theta)
        self.body_points = []
        self.wheels: List[
            Wheel
        ] = []  ## List of wheels, left to right, front to back 0,1,2,3
        for j in [1, -1]:
            for i in [-1, 1]:
                self.wheels.append(Wheel(x=i * self.width / 2, y=j * self.length / 2))
                self.body_points.append([i * self.width / 2, j * self.length / 2, 1])
        self.update_all_steer([0, 0, 0, 0])  # set all wheels to 0 steer
        self.body_points = np.array(self.body_points)
        self.update_transform(x, y, theta)

    def update_transform(self, x, y, theta):
        """update the transform matrix from world to vehicle frame

        Args:
            x (_type_): x in world frame
            y (_type_): y in world frame
            theta (_type_): theta in world frame
        """
        self.w_v_transform = transform_2d(x, y, theta)

    def update_all_steer(self, theta_list):
        """update all wheels steer angle

        Args:
            theta_list (_type_): _description_
        """
        for i in range(len(self.wheels)):
            self.wheels[i].update_steer(theta_list[i])

    def update_pose(self, x, y, theta):
        """update the pose of the car

        Args:
            x (_type_): x in world frame
            y (_type_): y in world frame
            theta (_type_): theta in world frame
        """
        self.x, self.y, self.theta = x, y, theta
        self.update_transform(x, y, theta)


if __name__ == "__main__":

    car = Car(x=5, y=5, theta=0, length=4, width=2)
    print(f"body points in car frame: \n{car.body_points}")

    world_points = np.dot(car.w_v_transform, car.body_points.T).T
    print(f"body points in world frame: \n{world_points}")
