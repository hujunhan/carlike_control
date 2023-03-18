# helper functions for carlike control
import numpy as np

def transform_2d(x, y, theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, x],
                     [s, c, y],
                     [0, 0, 1]])