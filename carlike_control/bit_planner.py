import numpy as np
import math
import random
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from carlike_control.env import Environment
from loguru import logger as log

# log.remove()


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None


class Tree:
    def __init__(self, x_start, x_goal):
        self.x_start = x_start  # start node
        self.goal = x_goal  # end node

        self.r = 4.0
        self.V = set()  # vertices of RGG
        self.E = set()  # edges of RGG
        self.QE = set()  # queue of edges of RGG
        self.QV = set()  # queue of vertices of RGG

        self.V_old = set()


class BITStar:
    def __init__(self, env: Environment, x_start, x_goal, eta, iter_max):
        self.x_start = Node(x_start[0], x_start[1])
        self.x_goal = Node(x_goal[0], x_goal[1])
        self.eta = eta
        self.iter_max = iter_max
        self.path_history = []
        self.env = env

        self.delta = self.env.delta
        self.x_range = [0.0, self.env.width]
        self.y_range = [0.0, self.env.height]

        self.Tree = Tree(self.x_start, self.x_goal)  # RGG tree
        self.X_sample = set()
        self.g_T = dict()  # upper bound of cost-to-come

    def init(self):
        self.Tree.V.add(self.x_start)  # the tree is grown from the start node
        self.X_sample.add(self.x_goal)

        self.g_T[self.x_start] = 0.0
        self.g_T[self.x_goal] = np.inf

        cMin, theta = self.calc_dist_and_angle(self.x_start, self.x_goal)
        C = self.RotationToWorldFrame(self.x_start, self.x_goal, cMin)
        xCenter = np.array(
            [
                [(self.x_start.x + self.x_goal.x) / 2.0],
                [(self.x_start.y + self.x_goal.y) / 2.0],
                [0.0],
            ]
        )

        return theta, cMin, xCenter, C

    def reset(self):
        self.Tree = Tree(self.x_start, self.x_goal)  # RGG tree
        self.X_sample = set()
        self.g_T = dict()  # upper bound of cost-to-come

    def planning(self):
        theta, cMin, xCenter, C = self.init()

        for k in range(self.iter_max):
            if not self.Tree.QE and not self.Tree.QV:
                if k == 0:
                    m = self.x_range[1] * self.y_range[1] / 5
                else:
                    m = self.x_range[1] * self.y_range[1] / 5

                if self.x_goal.parent is not None:
                    print(f"goal reached")
                    print(f"iteration: {k}")
                    path_x, path_y = self.ExtractPath()
                    return path_x, path_y
                    # print(f"path_x: {path_x}, path_y: {path_y}")
                    # self.path_history.append((path_x, path_y))

                self.Prune(self.g_T[self.x_goal])
                self.X_sample.update(
                    self.Sample(m, self.g_T[self.x_goal], cMin, xCenter, C)
                )
                self.Tree.V_old = {v for v in self.Tree.V}
                self.Tree.QV = {v for v in self.Tree.V}
                # self.Tree.r = self.radius(len(self.Tree.V) + len(self.X_sample))

            count = 0
            while self.BestVertexQueueValue() <= self.BestEdgeQueueValue():
                result = self.BestInVertexQueue()
                self.ExpandVertex(result)

            vm, xm = self.BestInEdgeQueue()
            self.Tree.QE.remove((vm, xm))

            if (
                self.g_T[vm] + self.calc_dist(vm, xm) + self.h_estimated(xm)
                < self.g_T[self.x_goal]
            ):
                log.debug(f"first condition")
                actual_cost = self.cost(vm, xm)
                if (
                    self.g_estimated(vm) + actual_cost + self.h_estimated(xm)
                    < self.g_T[self.x_goal]
                ):
                    log.debug(f"second condition")
                    if self.g_T[vm] + actual_cost < self.g_T[xm]:
                        log.debug(f"third condition")
                        if xm in self.Tree.V:
                            # remove edges
                            log.debug(f"remove edge")
                            edge_delete = set()
                            for v, x in self.Tree.E:
                                if x == xm:
                                    edge_delete.add((v, x))

                            for edge in edge_delete:
                                self.Tree.E.remove(edge)
                        else:
                            log.debug(f"add {xm.x}, {xm.y}")
                            self.X_sample.remove(xm)
                            self.Tree.V.add(xm)
                            self.Tree.QV.add(xm)

                        self.g_T[xm] = self.g_T[vm] + actual_cost
                        self.Tree.E.add((vm, xm))
                        xm.parent = vm

                        set_delete = set()
                        for v, x in self.Tree.QE:
                            if (
                                x == xm
                                and self.g_T[v] + self.calc_dist(v, xm) >= self.g_T[xm]
                            ):
                                set_delete.add((v, x))

                        for edge in set_delete:
                            self.Tree.QE.remove(edge)
            else:
                self.Tree.QE = set()
                self.Tree.QV = set()

            # if k % 5 == 0:
            #     self.animation(xCenter, self.g_T[self.x_goal], cMin, theta)

        path_x, path_y = self.ExtractPath()
        return path_x, path_y

    def ExtractPath(self):
        node = self.x_goal
        path_x, path_y = [node.x], [node.y]

        while node.parent:
            node = node.parent
            path_x.append(node.x)
            path_y.append(node.y)

        return path_x, path_y

    def Prune(self, cBest):
        self.X_sample = {x for x in self.X_sample if self.f_estimated(x) < cBest}
        self.Tree.V = {v for v in self.Tree.V if self.f_estimated(v) <= cBest}
        self.Tree.E = {
            (v, w)
            for v, w in self.Tree.E
            if self.f_estimated(v) <= cBest and self.f_estimated(w) <= cBest
        }
        self.X_sample.update({v for v in self.Tree.V if self.g_T[v] == np.inf})
        self.Tree.V = {v for v in self.Tree.V if self.g_T[v] < np.inf}

    def cost(self, start, end):
        if self.env.is_cross_obstacle(start.x, start.y, end.x, end.y):
            log.debug(f"cross obstacle")
            return np.inf

        return self.calc_dist(start, end)

    def f_estimated(self, node):
        return self.g_estimated(node) + self.h_estimated(node)

    def g_estimated(self, node):
        return self.calc_dist(self.x_start, node)

    def h_estimated(self, node):
        return self.calc_dist(node, self.x_goal)

    def Sample(self, m, cMax, cMin, xCenter, C):
        if cMax < np.inf:
            return self.SampleEllipsoid(m, cMax, cMin, xCenter, C)
        else:
            return self.SampleFreeSpace(m)

    def SampleEllipsoid(self, m, cMax, cMin, xCenter, C):
        r = [
            cMax / 2.0,
            math.sqrt(cMax**2 - cMin**2) / 2.0,
            math.sqrt(cMax**2 - cMin**2) / 2.0,
        ]
        L = np.diag(r)

        ind = 0
        delta = self.delta
        Sample = set()

        while ind < m:
            xBall = self.SampleUnitNBall()
            x_rand = np.dot(np.dot(C, L), xBall) + xCenter
            node = Node(x_rand[(0, 0)], x_rand[(1, 0)])
            in_obs = self.env.is_in_obstacle(node.x, node.y)
            in_x_range = self.x_range[0] + delta <= node.x <= self.x_range[1] - delta
            in_y_range = self.y_range[0] + delta <= node.y <= self.y_range[1] - delta

            if not in_obs and in_x_range and in_y_range:
                Sample.add(node)
                ind += 1
        # plot the sampled points
        # log.debug(f"sampleEllipsoid: {len(Sample)}")
        # plt.scatter([node.x for node in Sample], [node.y for node in Sample], s=1)
        # plt.show()
        return Sample

    def SampleFreeSpace(self, m):
        delta = self.delta
        Sample = set()

        ind = 0
        while ind < m:
            node = Node(
                random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                random.uniform(self.y_range[0] + delta, self.y_range[1] - delta),
            )
            if self.env.is_in_obstacle(node.x, node.y):
                continue
            else:
                Sample.add(node)
                ind += 1
        # plot the sampled points
        # plt.scatter([node.x for node in Sample], [node.y for node in Sample], s=1)
        # plt.show()
        return Sample

    def radius(self, q):
        cBest = self.g_T[self.x_goal]
        lambda_X = len([1 for v in self.Tree.V if self.f_estimated(v) <= cBest])
        radius = 2 * self.eta * (1.5 * lambda_X / math.pi * math.log(q) / q) ** 0.5

        return radius

    def ExpandVertex(self, v):
        self.Tree.QV.remove(v)
        X_near = {x for x in self.X_sample if self.calc_dist(x, v) <= self.Tree.r}

        for x in X_near:
            if (
                self.g_estimated(v) + self.calc_dist(v, x) + self.h_estimated(x)
                < self.g_T[self.x_goal]
            ):
                self.g_T[x] = np.inf
                self.Tree.QE.add((v, x))

        if v not in self.Tree.V_old:
            V_near = {w for w in self.Tree.V if self.calc_dist(w, v) <= self.Tree.r}

            for w in V_near:
                if (
                    (v, w) not in self.Tree.E
                    and self.g_estimated(v) + self.calc_dist(v, w) + self.h_estimated(w)
                    < self.g_T[self.x_goal]
                    and self.g_T[v] + self.calc_dist(v, w) < self.g_T[w]
                ):
                    self.Tree.QE.add((v, w))
                    if w not in self.g_T:
                        self.g_T[w] = np.inf

    def BestVertexQueueValue(self):
        if not self.Tree.QV:
            log.error(f"QV is Empty! in BestVertexQueueValue")
            return np.inf

        return min(self.g_T[v] + self.h_estimated(v) for v in self.Tree.QV)

    def BestEdgeQueueValue(self):
        if not self.Tree.QE:
            log.error(f"QE is Empty! in BestEdgeQueueValue")
            return np.inf

        return min(
            self.g_T[v] + self.calc_dist(v, x) + self.h_estimated(x)
            for v, x in self.Tree.QE
        )

    def BestInVertexQueue(self):
        if not self.Tree.QV:
            log.error("QV is Empty! in BestInVertexQueue")
            return None

        v_value = {v: self.g_T[v] + self.h_estimated(v) for v in self.Tree.QV}

        return min(v_value, key=v_value.get)

    def BestInEdgeQueue(self):
        if not self.Tree.QE:
            log.error("QE is Empty! in BestInEdgeQueue")
            return None

        e_value = {
            (v, x): self.g_T[v] + self.calc_dist(v, x) + self.h_estimated(x)
            for v, x in self.Tree.QE
        }

        return min(e_value, key=e_value.get)

    @staticmethod
    def SampleUnitNBall():
        while True:
            x, y = random.uniform(-1, 1), random.uniform(-1, 1)
            if x**2 + y**2 < 1:
                return np.array([[x], [y], [0.0]])

    @staticmethod
    def RotationToWorldFrame(x_start, x_goal, L):
        a1 = np.array(
            [[(x_goal.x - x_start.x) / L], [(x_goal.y - x_start.y) / L], [0.0]]
        )
        e1 = np.array([[1.0], [0.0], [0.0]])
        M = a1 @ e1.T
        U, _, V_T = np.linalg.svd(M, True, True)
        C = U @ np.diag([1.0, 1.0, np.linalg.det(U) * np.linalg.det(V_T.T)]) @ V_T

        return C

    @staticmethod
    def calc_dist(start, end):
        return math.hypot(start.x - end.x, start.y - end.y)

    @staticmethod
    def calc_dist_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)


if __name__ == "__main__":
    from carlike_control.viz import Visualization
    import matplotlib.pyplot as plt

    random.seed(1)

    np.random.seed(1)
    WIDTH = 90
    HEIGHT = 90
    viz = Visualization((-10, WIDTH + 10), (-10, HEIGHT + 10))

    x_start = (6, 86)  # Starting node
    x_goal = (80, 26)  # Goal node
    eta = 2
    iter_max = 1500
    log.debug("start!!!")

    env = Environment(WIDTH, HEIGHT, 50)
    while env.is_in_obstacle(x_start[0], x_start[1]):
        x_start = (random.uniform(0, WIDTH), random.uniform(0, HEIGHT))
    while env.is_in_obstacle(x_goal[0], x_goal[1]):
        x_goal = (random.uniform(0, WIDTH), random.uniform(0, HEIGHT))
    log.debug(f"x_start: {x_start}")
    log.debug(f"x_goal: {x_goal}")
    # env.add_obstacle(10, 6, "rectangle")

    bit = BITStar(env, x_start, x_goal, eta, iter_max)
    # bit.animation("Batch Informed Trees (BIT*)")
    import time

    a = time.time()
    path_x, path_y = bit.planning()
    b = time.time()
    print(f"elapsed time: {b-a}")
    viz.draw_env(env)

    viz.draw_path(path_x, path_y)
    plt.show()
