# Simulate the carlike vehicle model using bicycle model
# Ref: https://ftp.idu.ac.id/wp-content/uploads/ebook/tdg/TERRAMECHANICS%20AND%20MOBILITY/epdf.pub_vehicle-dynamics-and-control-2nd-edition.pdf

import numpy as np
import matplotlib.pyplot as plt

x = 0  # m # x position of the vehicle
y = 0  # m # y position of the vehicle

base_r = (
    1.5  # m # distance from the center of the rear axle to the center of the vehicle
)
base_f = (
    1.5  # m # distance from the center of the front axle to the center of the vehicle
)

v = 10  # m/s # linear velocity of the vehicle
# v_angle=np.pi/6 # rad # linear velocity angle in vehicle frame
steer_r = np.pi / 12  # m # steering angle of the rear wheel
steer_f = np.pi / 10  # m # steering angle of the front wheel

heading_angle = 0  # rad # heading angle of the vehicle

if __name__ == "__main__":
    update_rate = 100  # Hz
    interval = 1 / update_rate  # s
    simulate_time = 30  # s
    history = np.zeros((int(simulate_time / interval), 3))
    history[0, :] = [x, y, heading_angle]
    for i in range(1, int(simulate_time / interval)):
        v_angle = np.arctan(
            (base_r * np.tan(steer_f) + base_f * np.tan(steer_r)) / (base_r + base_f)
        )
        d_x = v * np.cos(heading_angle + v_angle) * interval
        d_y = v * np.sin(heading_angle + v_angle) * interval
        d_heading_angle = (
            v
            * np.cos(v_angle)
            / (base_r + base_f)
            * (np.tan(steer_f) - np.tan(steer_r))
            * interval
        )

        x = x + d_x
        y = y + d_y
        heading_angle = heading_angle + d_heading_angle
        history[i, :] = [x, y, heading_angle]

    # downsample the history
    history = history[::20, :]

    # plot the car trajectory using dot using matplotlib
    plt.scatter(history[:, 0], history[:, 1], s=1)
    # plot the car heading angle using arrow using matplotlib
    plt.quiver(
        history[:, 0],
        history[:, 1],
        np.cos(history[:, 2]),
        np.sin(history[:, 2]),
        scale=10,
    )
    # equal aspect ratio
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()
