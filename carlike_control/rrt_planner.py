import math
import random

import matplotlib.pyplot as plt
import numpy as np
from carlike_control.pgm import Environment


class RRT:
    class Node:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.parent = None

    def __init__(self, env: Environment, start, goal, step=0.1) -> None:
        self.env = env
        self.start = self.Node(start[0], start[1])
        self.goal = self.Node(goal[0], goal[1])
        self.step = step
        self.node_list = []

    def planning(self):
        self.node_list.append(self.start)
        count = 0
        while True:
            print(count)
            count += 1
            rnd_node = self.get_random_node()
            nearest_node = self.get_nearest_node(rnd_node)
            new_node = self.steer(nearest_node, rnd_node, self.step)
            if self.env.is_in_obstacle(new_node.x, new_node.y):
                continue
            self.node_list.append(new_node)
            if self.is_goal(new_node):
                print("Goal!!")
                break
        return self.get_path(new_node)

    def get_path(self, node: Node):
        path = [[node.x, node.y]]
        while node.parent is not None:
            node = node.parent
            path.append([node.x, node.y])
        return path

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = self.Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)
        new_node.parent = from_node
        if extend_length > d:
            extend_length = d
        new_node.x += extend_length * math.cos(theta)
        new_node.y += extend_length * math.sin(theta)
        return new_node

    def calc_distance_and_angle(self, from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.sqrt(dx**2 + dy**2)
        theta = math.atan2(dy, dx)
        return d, theta

    def get_random_node(self):
        if random.randint(0, 100) > 20:
            rnd = self.Node(
                random.uniform(0, self.env.width),
                random.uniform(0, self.env.height),
            )
        else:  # goal point sampling
            rnd = self.Node(self.goal.x, self.goal.y)
        return rnd

    def get_nearest_node(self, rnd_node):
        dlist = [
            (node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y) ** 2
            for node in self.node_list
        ]
        minind = dlist.index(min(dlist))
        return self.node_list[minind]

    def is_goal(self, node: Node):
        return (node.x - self.goal.x) ** 2 + (node.y - self.goal.y) ** 2 < 0.1**2
