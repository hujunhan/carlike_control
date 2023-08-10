from carlike_control.rrt_planner import RRT
from carlike_control.pgm import Environment
import matplotlib.pyplot as plt
import numpy as np

env = Environment("./data/map.pgm", "./data/map.yaml")
x_start = (86, 205)  # Starting node
x_goal = (468, 306)  # Goal node
print(f"x_start: {x_start}\nx_goal: {x_goal}")
round_max = 10
iter_max = 2000
step = 5
rrt = RRT(env, x_start, x_goal, step=step)
path = rrt.planning(iter_max=iter_max, round_max=round_max)
print(f"RRT path: {path}")

map_with_path = np.copy(env.map)
path = rrt.smooth_path(path)
path = (np.asarray(path)).astype(int)
path_x = path[:, 0]
path_y = path[:, 1]
plt.imshow(map_with_path, cmap="gray")
# plt.scatter(path_x, path_y, c="r", s=1)
# draw path by line
plt.plot(path_x, path_y, c="r", linewidth=1)
plt.show()
