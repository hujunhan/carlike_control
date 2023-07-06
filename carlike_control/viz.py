# Visualization class for carlike_control
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle, Circle

from carlike_control.car import Car
from carlike_control.wheel import Wheel
from carlike_control.helper import transform_2d
from carlike_control.env import Environment, Obstacle


class Visualization:
    def __init__(self, xlim=(-10, 10), ylim=(-10, 10)) -> None:
        """Visualization class for plotting the car and wheels

        Args:
            xlim (tuple, optional): canvas limit x. Defaults to (-10,10).
            ylim (tuple, optional): canvas limit y. Defaults to (-10,10).
        """
        self.fig, self.ax = plt.subplots()
        self.xlim = xlim
        self.ylim = ylim
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_aspect("equal")
        self.show_order_index = [0, 1, 3, 2]
        pass

    def draw_via_points(self, via_points):
        for p in via_points:
            self.ax.plot(p[0], p[1], "x", color="red")

    def draw_path(self, path_x, path_y):
        self.ax.plot(path_x, path_y, "-", color="green")
        pass

    def draw_car(self, car: Car):
        """Draw the car and wheels

        Args:
            car (Car): input car
        """
        w_v_transform = transform_2d(car.x, car.y, car.yaw)
        world_points = np.dot(w_v_transform, car.body_points.T).T
        # print(f'body points in world frame: \n{world_points}')
        # Draw the car
        self.ax.add_patch(
            Polygon(world_points[self.show_order_index, :2], color="blue")
        )
        # Draw the wheels
        for wheel in car.wheels:
            self.draw_wheel(wheel, w_v_transform)
        pass

    def draw_wheel(self, wheel: Wheel, w_v_transform):
        ## Draw the wheel
        # Transform the points to the vehicle frame
        v_c_transform = transform_2d(wheel.x, wheel.y, wheel.yaw)
        vehicle_points = np.dot(v_c_transform, wheel.body_points.T).T
        # Transform the points to the world frame
        world_points = np.dot(w_v_transform, vehicle_points.T).T
        # Draw the wheel
        self.ax.add_patch(Polygon(world_points[self.show_order_index, :2], color="red"))
        pass

    def clear(self):
        ## Clear the visualization
        self.ax.clear()
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        self.ax.set_aspect("equal")
        pass

    def draw_env(self, env: Environment):
        for obstacle in env.obstacles:
            self.draw_obstacle(obstacle)
        self.draw_boundary(env)

    def draw_obstacle(self, obstacle: Obstacle):
        if obstacle.shape == "rectangle":
            self.ax.add_patch(
                Rectangle(
                    (obstacle.x - obstacle.width / 2, obstacle.y - obstacle.height / 2),
                    obstacle.width,
                    obstacle.height,
                    color="black",
                )
            )
        elif obstacle.shape == "circle":
            self.ax.add_patch(
                Circle((obstacle.x, obstacle.y), obstacle.radius, color="black")
            )
        pass

    def draw_boundary(self, env: Environment):
        self.ax.add_patch(
            Rectangle(
                (0, 0),
                env.width,
                env.height,
                color="black",
                fill=False,
            )
        )
        pass


if __name__ == "__main__":
    viz = Visualization((0, 10), (0, 10))
    car = Car(x=5, y=5, yaw=0, length=4, width=2)
    car.update_all_steer([np.pi / 6, np.pi / 6, 0, 0])
    viz.draw_car(car)
    plt.show()
    car.update_pose(3, 2, np.pi / 3)
    viz.draw_car(car)
    print("update car pose")
