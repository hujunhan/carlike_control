import numpy as np
from carlike_control.env import Environment, Obstacle
from carlike_control.viz import Visualization
from matplotlib import pyplot as plt

np.random.seed(0)
WIDTH = 100
HEIGHT = 100
OBSTACLE_NUM = 10
env = Environment(WIDTH, HEIGHT, OBSTACLE_NUM)

viz = Visualization((-10, WIDTH + 10), (-10, HEIGHT + 10))

viz.draw_env(env)
print(env.is_in_obstacle(14, 21))
print(env.is_in_obstacle(35, 23))
print(env.is_cross_obstacle(6, 30, 23, 14))
print(env.is_cross_obstacle(80, 58, 90, 65))

plt.show()
