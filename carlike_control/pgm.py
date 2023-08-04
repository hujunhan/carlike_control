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
        self.delta = 0.2  # grace period for collision detection
        # read the map into a numpy array
        self.map = self.read_pgm(mpg_path, byteorder="<")
        # read the yaml file into setting
        with open(yaml_path, "r") as stream:
            try:
                self.setting = yaml.safe_load(stream)
                self.resolution = self.setting["resolution"]
                self.width = self.map.shape[1] * self.resolution
                self.height = self.map.shape[0] * self.resolution
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
        self.enlarged_map = np.ones_like(self.map) * 255
        delta_pixel = int(self.delta / self.resolution)
        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                if self.map[i, j] < 250:
                    self.enlarged_map[
                        max(0, i - delta_pixel) : min(
                            i + delta_pixel, self.map.shape[0]
                        ),
                        max(0, j - delta_pixel) : min(
                            j + delta_pixel, self.map.shape[1]
                        ),
                    ] = 0

    def is_in_obstacle(self, x, y):
        x = int(x / self.resolution)
        y = int(y / self.resolution)
        data = self.enlarged_map[y, x]
        if data < 128:
            return True
        else:
            return False

    def is_cross_obstacle(self, x1, y1, x2, y2):
        """check if the line (x1, y1) -> (x2, y2) cross the obstacle
        examine every obstacle to see if the line cross it
        Args:
            x1 (_type_): _description_
            y1 (_type_): _description_
            x2 (_type_): _description_
            y2 (_type_): _description_

        Returns:
            _type_: _description_
        """
        x1 = int(x1 / self.resolution)
        y1 = int(y1 / self.resolution)
        x2 = int(x2 / self.resolution)
        y2 = int(y2 / self.resolution)
        x_step = 1 if x1 < x2 else -1
        y_step = 1 if y1 < y2 else -1
        # check if every point in the line is in the obstacle
        for x in range(x1, x2, x_step):
            for y in range(y1, y2, y_step):
                if self.is_in_obstacle(x * self.resolution, y * self.resolution):
                    return True


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
