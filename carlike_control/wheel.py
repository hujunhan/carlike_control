## Wheel class for carlike_control

import numpy as np
from carlike_control.helper import transform_2d


class Wheel:
    def __init__(
        self, x=0, y=0, yaw=0, radius=0.1, width=0.065, limit=[-np.pi / 2, np.pi / 2]
    ) -> None:
        self.x = x  # position of the wheel in the vehicle frame
        self.y = y
        self.radius = radius
        self.width = width
        self.yaw = yaw
        self.body_points = []
        for i in [-1, 1]:
            for j in [-1, 1]:
                self.body_points.append([i * self.radius, j * self.width / 2, 1])
        self.body_points = np.array(self.body_points)
        pass

    def update_steer(self, yaw):
        self.yaw = yaw


if __name__ == "__main__":
    wheel = Wheel(1, 2, 0, 0.5, 0.2)
    print(f"body points in wheel frame: \n{wheel.body_points}")

    vehicle_points = np.dot(wheel.v_c_transform, wheel.body_points.T).T
    print(f"body points in vehicle frame: \n{vehicle_points}")
