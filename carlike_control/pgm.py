## Script to create the environment for the carlike robot
# include Environment class
# it will read a PGM file and related yaml file to create the environment
# the PGM file is a map of the environment, the yaml file contains the
# resolution of the map, the origin of the map, and the obstacle positions
import numpy as np
from typing import List
import math
import yaml
import re
from scipy.signal import convolve2d


class Environment:
    def read_pgm(self, filename, byteorder=">"):
        """Return image data from a raw PGM file as numpy array.

        Format specification: http://netpbm.sourceforge.net/doc/pgm.html

        """
        with open(filename, "rb") as f:
            buffer = f.read()
        try:
            header, width, height, maxval = re.search(
                b"(^P5\s(?:\s*#.*[\r\n])*"
                b"(\d+)\s(?:\s*#.*[\r\n])*"
                b"(\d+)\s(?:\s*#.*[\r\n])*"
                b"(\d+)\s(?:\s*#.*[\r\n]\s)*)",
                buffer,
            ).groups()
        except AttributeError:
            raise ValueError("Not a raw PGM file: '%s'" % filename)
        return np.frombuffer(
            buffer,
            dtype="u1" if int(maxval) < 256 else byteorder + "u2",
            count=int(width) * int(height),
            offset=len(header),
        ).reshape((int(height), int(width)))

    def __init__(self, mpg_path, yaml_path):
        self.delta = 0.1  # grace period for collision detection
        # read the map into a numpy array
        self.map = self.read_pgm(mpg_path, byteorder="<")
        # read the yaml file into setting
        with open(yaml_path, "r") as stream:
            try:
                self.setting = yaml.safe_load(stream)
                self.resolution = self.setting["resolution"]
                self.width = self.map.shape[1] * self.resolution
                self.height = self.map.shape[0] * self.resolution
                self.digit_width = self.map.shape[1]
                self.digit_height = self.map.shape[0]
                self.origin = self.setting["origin"]
                self.occupied_thresh = self.setting["occupied_thresh"] * 255
                self.free_thresh = self.setting["free_thresh"] * 255
                print(self.setting)
            except yaml.YAMLError as exc:
                print(exc)
        self.generate_enlarge_map()

    def generate_enlarge_map(self):
        # enlarge the obstacle by delta
        # all the obstacle arounded by delta will be considered as obstacle
        temp_map = np.ones(self.map.shape, dtype=np.uint8) * 255  # obstacle map
        temp_map[self.map > 250] = 0  # moveable area
        kernel_size = int(self.delta / self.resolution) * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.enlarged_map = convolve2d(
            temp_map, kernel, mode="same", boundary="fill", fillvalue=255
        )
        # self.enlarged_map[self.enlarged_map > 0] = 255
        # generate the obstacle map, true/false
        self.in_obstacle_map = np.ones(self.map.shape, dtype=np.uint8)
        self.in_obstacle_map[self.enlarged_map == 0] = 0  # moveable area

    def is_in_obstacle(self, x, y):
        # print(f"x: {x}, y: {y}, in obstacle: {self.in_obstacle_map[y, x]}")
        return self.in_obstacle_map[y, x] == 1

    def is_cross_obstacle(self, x1, y1, x2, y2):
        """check if the line (x1, y1) -> (x2, y2) cross the obstacle
        examine every obstacle to see if the line cross it
        Returns:
            _type_: _description_
        """
        # check if every point in the line is in the obstacle
        for x, y in self.bresenham_line(x1, y1, x2, y2):
            if self.is_in_obstacle(x, y):
                return True
        return False

    def bresenham_line(self, x1, y1, x2, y2):
        """Generate points between (x1, y1) and (x2, y2) using Bresenham's line algorithm"""
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        if dx > dy:
            err = dx / 2.0
            while x1 != x2:
                points.append((x1, y1))
                err -= dy
                if err < 0:
                    y1 += sy
                    err += dx
                x1 += sx
        else:
            err = dy / 2.0
            while y1 != y2:
                points.append((x1, y1))
                err -= dx
                if err < 0:
                    x1 += sx
                    err += dy
                y1 += sy
        points.append((x1, y1))
        return points


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    env = Environment("./data/map.pgm", "./data/map.yaml")
    # draw the map

    # the size of the map
    width = env.map.shape[1] * env.resolution
    height = env.map.shape[0] * env.resolution
    print(width, height)
    plt.imshow(env.map, cmap="gray")
    plt.show()
    plt.imshow(env.enlarged_map, cmap="gray")
    plt.show()
    plt.imshow(env.in_obstacle_map, cmap="gray")
    plt.show()
