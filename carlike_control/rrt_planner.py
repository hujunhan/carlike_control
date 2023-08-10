import math
import random

import matplotlib.pyplot as plt
import numpy as np
from carlike_control.pgm import Environment
from scipy.spatial import KDTree


class RRT:
    class Node:
        def __init__(self, x, y, parent=None):
            self.x = x
            self.y = y
            self.parent = parent

    def __init__(self, env: Environment, start, goal, step=1) -> None:
        self.env = env
        self.start = self.Node(start[0], start[1])
        self.goal = self.Node(goal[0], goal[1])
        self.step = step
        self.node_list = []
        self.node_xy_list = []  # store the node pos  for fast search
        self.kdtree = KDTree([[self.start.x, self.start.y]])

    def planning(self, round_max=5, iter_max=5000):
        round = 0
        while round < round_max:
            self.node_list = [self.start]
            self.node_xy_list = [[self.start.x, self.start.y]]
            self.kdtree = KDTree([[self.start.x, self.start.y]])
            count = 0
            while count < iter_max:
                # print(count)
                rnd_node = self.get_random_node()  # a node in moveable space or goal
                nearest_node = self.get_nearest_node(rnd_node)
                if nearest_node.x == rnd_node.x and nearest_node.y == rnd_node.y:
                    continue
                new_node = self.steer(nearest_node, rnd_node, self.step)
                if self.env.is_in_obstacle(
                    new_node.x, new_node.y
                ) or self.env.is_cross_obstacle(
                    nearest_node.x, nearest_node.y, new_node.x, new_node.y
                ):
                    continue
                count += 1
                if count % 100 == 0:
                    print(f"round: {round}, iter: {count}")
                self.node_list.append(new_node)
                self.node_xy_list.append([new_node.x, new_node.y])
                self.kdtree = KDTree(self.node_xy_list)
                if self.is_goal(new_node):
                    print(
                        f"Goal!! {round} rounds, {count} iterations, {len(self.node_list)} nodes"
                    )
                    return self.get_path(new_node)
            round += 1

    def get_path(self, node: Node):
        path = [[node.x, node.y]]
        while node.parent is not None:
            node = node.parent
            path.append([node.x, node.y])
        return path

    def steer(self, from_node, to_node, extend_length=1):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        distance = math.sqrt(dx**2 + dy**2)
        if extend_length > distance:
            extend_length = distance
        dx = math.ceil(dx * extend_length / distance)
        dy = math.ceil(dy * extend_length / distance)
        return self.Node(from_node.x + dx, from_node.y + dy, parent=from_node)

    def get_random_node(self):
        if random.random() > 0.20:
            rnd_x = random.randint(1, self.env.digit_width - 1)
            rnd_y = random.randint(1, self.env.digit_height - 1)
            rnd = self.Node(rnd_x, rnd_y)
        else:  # goal point sampling
            rnd = self.Node(self.goal.x, self.goal.y)
        return rnd

    def get_nearest_node(self, rnd_node):
        # nearby_nodes = self.get_nearby_nodes(rnd_node, radius=200)
        # if not nearby_nodes:
        #     nodes = np.array([[node.x, node.y] for node in self.node_list])
        #     distances = np.sum((nodes - [rnd_node.x, rnd_node.y]) ** 2, axis=1)
        #     return self.node_list[np.argmin(distances)]

        # distances = []
        # for node in nearby_nodes:
        #     d = (node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y) ** 2
        #     distances.append(d)
        # min_index = np.argmin(distances)
        # return nearby_nodes[min_index]
        distance, index = self.kdtree.query([(rnd_node.x, rnd_node.y)])
        return self.node_list[index[0]]

    def get_nearby_nodes(self, rnd_node, radius=3):
        # nodes = np.array(self.node_xy_list)
        nodes = np.array([[node.x, node.y] for node in self.node_list])
        distances = np.sum((nodes - [rnd_node.x, rnd_node.y]) ** 2, axis=1)
        nearby_indices = np.where(distances <= radius**2)[0]
        return [self.node_list[i] for i in nearby_indices]

    def is_goal(self, node: Node):
        return node.x == self.goal.x and node.y == self.goal.y

    def smooth_path(self, path):
        i = 0
        while i < len(path) - 2:
            node1 = path[i]
            node2 = path[i + 2]
            # print(node1, node2)
            if not self.env.is_cross_obstacle(
                node1[0], node1[1], node2[0], node2[1]
            ):  # Assuming your env supports line checks
                del path[i + 1]
            else:
                i += 1
        return path
