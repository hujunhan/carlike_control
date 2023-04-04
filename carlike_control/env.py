## Script to create the environment for the carlike robot
# include Obstacle class and Environment class
import numpy as np


class Obstacle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        ## velocity of the obstacle, vector
        self.v = np.array([0, 0])
        self.shape = None

    def rectangle(self, width, height):
        self.width = width
        self.height = height
        self.shape = "rectangle"

    def circle(self, radius):
        self.radius = radius
        self.shape = "circle"

    def update(self, dt):
        self.x += self.v[0] * dt
        self.y += self.v[1] * dt


class Environment:
    def __init__(self, width, height, num_obstacles):
        self.width = width
        self.height = height
        self.obstacles = []

        # Add random obstacles
        for i in range(num_obstacles):
            x = np.random.uniform(0, self.width)
            y = np.random.uniform(0, self.height)
            shape = np.random.choice(["rectangle", "circle"])
            self.add_obstacle(x, y, shape)

    def add_obstacle(self, x, y, shape):
        obstacle = Obstacle(x, y)

        if shape == "rectangle":
            # add random width and height rectangle
            obstacle.rectangle(np.random.uniform(2, 5), np.random.uniform(2, 5))
        elif shape == "circle":
            # add random radius circle
            obstacle.circle(np.random.uniform(2, 5))
        self.obstacles.append(obstacle)
