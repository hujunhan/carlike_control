# helper functions for carlike control
import numpy as np
import math


def transform_2d(x, y, theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, x], [s, c, y], [0, 0, 1]])


def calc_beta(steer_front, steer_rear, l_f, l_r):
    return math.atan2(
        (l_f * np.tan(steer_rear) + l_r * np.tan(steer_front)), (l_f + l_r)
    )


def get_linear_model_matrix(
    v: float, yaw: float, steer_front: float, steer_rear: float, DT=0.1
):
    beta = calc_beta(steer_front, steer_rear)
    A = np.identity(NX)
    A[0, 2] = DT * math.cos(yaw + beta)
    A[0, 3] = -DT * v * math.sin(yaw + beta)
    A[1, 2] = DT * math.sin(yaw + beta)
    A[1, 3] = DT * v * math.cos(yaw + beta)
    A[3, 2] = (
        DT * np.cos(beta) * (np.tan(steer_front) - np.tan(steer_rear)) / (l_r + l_f)
    )

    B = np.zeros((NX, NU))
    B[2, 0] = DT
    B[3, 1] = DT * v * np.cos(beta) / (l_r + l_f) / np.cos(steer_front) ** 2
    B[3, 2] = -DT * v * np.cos(beta) / (l_r + l_f) / np.cos(steer_rear) ** 2

    C = np.zeros(NX)
    C[0] = DT * v * math.sin(yaw + beta) * yaw
    C[1] = -DT * v * math.cos(yaw + beta) * yaw
    C[3] = (
        -DT
        * v
        * np.cos(beta)
        / (l_r + l_f)
        * (1 / np.cos(steer_front) ** 2 - 1 / np.cos(steer_rear) ** 2)
    )
    return A, B, C
