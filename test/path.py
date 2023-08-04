from carlike_control.bit_planner import BITStar
from carlike_control.rrt_planner import RRT
from carlike_control.pgm import Environment
import matplotlib.pyplot as plt
import numpy as np

env = Environment("./data/map.pgm", "./data/map.yaml")
x_start = (86 * 0.05, 205 * 0.05)  # Starting node
x_goal = (616 * 0.05, 170 * 0.05)  # Goal node
print(f"x_start: {x_start}\nx_goal: {x_goal}")
iter_max = 1500
rrt = RRT(env, x_start, x_goal, 0.1)
path = rrt.planning()
print(f"RRT path: {path}")

map_with_path = np.copy(env.map)
path = (np.asarray(path) / env.resolution).astype(int)
path_x = path[:, 0]
path_y = path[:, 1]
plt.imshow(map_with_path, cmap="gray")
plt.scatter(path_x, path_y, c="r", s=1)
plt.show()
