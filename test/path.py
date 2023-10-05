from carlike_control.rrt_planner import RRT
from carlike_control.pgm import Environment
import matplotlib.pyplot as plt
import numpy as np

env = Environment("./data/company-1.pgm", "./data/company.yaml")
x_start = (20, 20)  # Starting node
x_goal = (280, 70)  # Goal node
print(f"x_start: {x_start}\nx_goal: {x_goal}")
round_max = (
    10  # Maximum number of iterations, 10 is enough, 因为每次都是随机的，所以多次迭代，可以确保找到一条路径
)
iter_max = (
    5000  # Maximum number of nodes to be added to the tree in each iteration 地图越大，这个值越大
)
step = 2  # Step size 每次走的步长，越大，路径越直，但是容易碰到障碍物
rrt = RRT(env, x_start, x_goal, step=step)
path = rrt.planning(iter_max=iter_max, round_max=round_max)
print(f"RRT path: {path}")

map_with_path = np.copy(env.map)
path = rrt.smooth_path(path)
path = (np.asarray(path)).astype(int)
path_x = path[:, 0]
path_y = path[:, 1]
print(path.flatten())
plt.imshow(map_with_path, cmap="gray")
# plt.scatter(path_x, path_y, c="r", s=1)
# draw path by line
plt.plot(path_x, path_y, c="r", linewidth=1)
plt.show()
