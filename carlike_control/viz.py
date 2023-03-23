# Visualization class for carlike_control
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from carlike_control.car import Car
from carlike_control.wheel import Wheel


class Visualization:
    def __init__(self, xlim=(-10, 10), ylim=(-10, 10)) -> None:
        """Visualization class for plotting the car and wheels

        Args:
            xlim (tuple, optional): canvas limit x. Defaults to (-10,10).
            ylim (tuple, optional): canvas limit y. Defaults to (-10,10).
        """
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_aspect("equal")
        self.show_order_index = [0, 1, 3, 2]
        pass

    def draw_car(self, car: Car):
        """Draw the car and wheels

        Args:
            car (Car): input car
        """
        world_points = np.dot(car.w_v_transform, car.body_points.T).T
        # print(f'body points in world frame: \n{world_points}')
        # Draw the car
        self.ax.add_patch(
            Polygon(world_points[self.show_order_index, :2], color="blue")
        )
        # Draw the wheels
        for wheel in car.wheels:
            self.draw_wheel(wheel, car.w_v_transform)
        pass

    def draw_wheel(self, wheel: Wheel, w_v_transform):
        ## Draw the wheel
        # Transform the points to the vehicle frame
        vehicle_points = np.dot(wheel.v_c_transform, wheel.body_points.T).T
        # Transform the points to the world frame
        world_points = np.dot(w_v_transform, vehicle_points.T).T
        # Draw the wheel
        self.ax.add_patch(Polygon(world_points[self.show_order_index, :2], color="red"))
        pass

    def show(self):
        ## Show the visualization
        plt.show()
        pass

    def clear(self):
        ## Clear the visualization
        self.ax.clear()
        pass


if __name__ == "__main__":
    viz = Visualization((-10, 10), (-10, 10))
    car = Car(x=5, y=5, theta=0, length=4, width=2)
    car.update_all_steer([np.pi / 5, np.pi / 5, np.pi / 3, np.pi / 6])
    viz.draw_car(car)
    # viz.show()

    car.update_pose(-3, 2, np.pi / 3)
    viz.draw_car(car)
    print("update car pose")
    viz.show()
