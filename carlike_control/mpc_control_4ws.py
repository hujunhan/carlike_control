## use MPC to control the carlike vehicle
# Ref: https://atsushisakai.github.io/PythonRobotics/modules/path_tracking/model_predictive_speed_and_steering_control/model_predictive_speed_and_steering_control.html

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from carlike_control.path_planning import calc_spline_course
import math
from loguru import logger
import cvxpy

# set logger only show the time without date, remain the color
import sys

logger.remove()
logger.add(sys.stderr, format="{time:HH:mm:ss} {message}", colorize=True)
NX = 4  # x = x, y, v, yaw
NU = 3  # a = [accel, steer_front, steer_rear]
T = 5  # horizon length
## Parameters
TARGET_SPEED = 3.6
MAX_TIME = 10.0
N_IND_SEARCH = 10  # Search index number
DT = 0.2
MAX_STEER = np.deg2rad(45.0)  # maximum steering angle [rad]
MAX_DSTEER = np.deg2rad(30.0)  # maximum steering speed [rad/s]
MAX_SPEED = 55.0 / 3.6  # maximum speed [m/s]
MIN_SPEED = -20.0 / 3.6  # minimum speed [m/s]
MAX_ACCEL = 1.0  # maximum accel [m/ss]
WB = 2.5  # [m]
l_f = 1.2  # [m] # distance from front wheel to center of gravity
l_r = 1.2  # [m] # distance from rear wheel to center of gravity
# iterative paramter
# mpc parameters
R = np.diag([0.01, 0.005, 0.005])  # input cost matrix
Rd = np.diag([0.01, 1.0, 1.0])


Q = np.diag([5.0, 5.0, 0.5, 0.0])  # state cost matrix

## convert R and Q from numpy array to list

Qf = Q  # state final matrix
GOAL_DIS = 1.5  # goal distance
STOP_SPEED = 0.5 / 3.6  # stop speed
MAX_ITER = 3  # Max iteration
DU_TH = 0.1  # iteration finish param


def pi_2_pi(angle):
    while angle > math.pi:
        angle = angle - 2.0 * math.pi

    while angle < -math.pi:
        angle = angle + 2.0 * math.pi

    return angle


class State:
    """
    vehicle state class
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.predelta = None


# Path definition
path = np.array([[0, 0], [10, 5], [20, 10], [30, 5], [40, 20]])


def get_path_course(path=path):
    x = path[:, 0]
    y = path[:, 1]
    ds = 1.0
    rx, ry, ryaw, rk, s = calc_spline_course(x, y, ds)
    return rx, ry, ryaw, rk, s


def calc_beta(steer_front, steer_rear):
    return math.atan2(
        (l_f * np.tan(steer_rear) + l_r * np.tan(steer_front)), (l_f + l_r)
    )


def calc_speed_profile(cx, cy, cyaw, target_speed):

    speed_profile = [target_speed] * len(cx)
    direction = 1.0  # forward

    # Set stop point
    for i in range(len(cx) - 1):
        dx = cx[i + 1] - cx[i]
        dy = cy[i + 1] - cy[i]

        move_direction = math.atan2(dy, dx)

        if dx != 0.0 and dy != 0.0:
            dangle = abs(pi_2_pi(move_direction - cyaw[i]))
            if dangle >= math.pi / 4.0:
                direction = -1.0
            else:
                direction = 1.0

        if direction != 1.0:
            speed_profile[i] = -target_speed
        else:
            speed_profile[i] = target_speed

    speed_profile[-1] = 0.0

    return speed_profile


def calc_nearest_index(state, cx, cy, cyaw, pind):
    """just find the nearest point in the path

    Args:
        state (_type_): _description_
        cx (_type_): _description_
        cy (_type_): _description_
        cyaw (_type_): _description_
        pind (_type_): _description_

    Returns:
        _type_: _description_
    """
    dx = [state.x - icx for icx in cx[pind : (pind + N_IND_SEARCH)]]
    dy = [state.y - icy for icy in cy[pind : (pind + N_IND_SEARCH)]]

    d = [idx**2 + idy**2 for (idx, idy) in zip(dx, dy)]

    mind = min(d)

    ind = d.index(mind) + pind

    mind = math.sqrt(mind)

    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y

    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        mind *= -1

    return ind, mind


def smooth_yaw(yaw):

    for i in range(len(yaw) - 1):
        dyaw = yaw[i + 1] - yaw[i]

        while dyaw >= math.pi / 2.0:
            yaw[i + 1] -= math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

        while dyaw <= -math.pi / 2.0:
            yaw[i + 1] += math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

    return yaw


def calc_ref_trajectory(state, cx, cy, cyaw, ck, sp, dl, pind):
    """get the reference trajectory, the next few points in the path,
    related to the current speed, get the reference path properties like
    x,y,speed,yaw

    Args:
        state (_type_): _description_
        cx (_type_): _description_
        cy (_type_): _description_
        cyaw (_type_): _description_
        ck (_type_): _description_
        sp (_type_): _description_
        dl (_type_): _description_
        pind (_type_): _description_

    Returns:
        _type_: _description_
    """
    xref = np.zeros((NX, T + 1))
    dref = np.zeros((2, T + 1))
    ncourse = len(cx)

    ind, _ = calc_nearest_index(state, cx, cy, cyaw, pind)

    if pind >= ind:
        ind = pind

    xref[0, 0] = cx[ind]
    xref[1, 0] = cy[ind]
    xref[2, 0] = sp[ind]
    xref[3, 0] = cyaw[ind]
    dref[0, 0] = 0.0  # steer operational point should be 0
    dref[1, 0] = 0.0  # steer operational point should be 0

    travel = 0.0

    for i in range(T + 1):
        travel += abs(state.v) * DT
        dind = int(round(travel / dl))

        if (ind + dind) < ncourse:
            xref[0, i] = cx[ind + dind]
            xref[1, i] = cy[ind + dind]
            xref[2, i] = sp[ind + dind]
            xref[3, i] = cyaw[ind + dind]
            dref[0, i] = 0.0
            dref[1, i] = 0.0
        else:
            xref[0, i] = cx[ncourse - 1]
            xref[1, i] = cy[ncourse - 1]
            xref[2, i] = sp[ncourse - 1]
            xref[3, i] = cyaw[ncourse - 1]
            dref[0, i] = 0.0
            dref[1, i] = 0.0

    return xref, ind, dref


def update_state(state: State, a, steer_front, steer_rear):
    """update the state of the vehicle according to the math model

    Args:
        state (State): _description_
        a (_type_): _description_
        delta (_type_): _description_

    Returns:
        _type_: _description_
    """
    # input check for steer
    steer_rear = np.clip(steer_rear, -MAX_STEER, MAX_STEER)
    steer_front = np.clip(steer_front, -MAX_STEER, MAX_STEER)
    beta = calc_beta(steer_front, steer_rear)

    state.x = state.x + state.v * math.cos(state.yaw + beta) * DT
    state.y = state.y + state.v * math.sin(state.yaw + beta) * DT
    state.yaw = (
        state.yaw
        + state.v
        * np.cos(beta)
        * (np.tan(steer_front) - np.tan(steer_rear))
        / (l_r + l_f)
        * DT
    )
    state.v = state.v + a * DT

    state.v = np.clip(state.v, MIN_SPEED, MAX_SPEED)

    return state


def predict_motion(x0, oa, od_f, od_r, xref):
    xbar = xref * 0.0
    for i, _ in enumerate(x0):
        xbar[i, 0] = x0[i]

    state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
    for (ai, di_f, di_r, i) in zip(oa, od_f, od_r, range(1, T + 1)):
        state = update_state(state, ai, di_f, di_r)
        xbar[0, i] = state.x
        xbar[1, i] = state.y
        xbar[2, i] = state.v
        xbar[3, i] = state.yaw

    return xbar


def get_nparray_from_matrix(x):
    return np.array(x).flatten()


def get_linear_model_matrix(v, phi, steer_front, steer_rear):
    beta = calc_beta(steer_front, steer_rear)
    A = np.zeros((NX, NX))
    A[0, 0] = 1.0
    A[1, 1] = 1.0
    A[2, 2] = 1.0
    A[3, 3] = 1.0
    A[0, 2] = DT * math.cos(phi + beta)
    A[0, 3] = -DT * v * math.sin(phi + beta)
    A[1, 2] = DT * math.sin(phi + beta)
    A[1, 3] = DT * v * math.cos(phi + beta)
    A[3, 2] = (
        DT * np.cos(beta) * (np.tan(steer_front) - np.tan(steer_rear)) / (l_r + l_f)
    )

    B = np.zeros((NX, NU))
    B[2, 0] = DT
    B[3, 1] = DT * v * np.cos(beta) / (l_r + l_f) / np.cos(steer_front) ** 2
    B[3, 2] = DT * v * np.cos(beta) / (l_r + l_f) / np.cos(steer_rear) ** 2

    C = np.zeros(NX)
    C[0] = DT * v * math.sin(phi + beta) * phi
    C[1] = -DT * v * math.cos(phi + beta) * phi
    C[3] = (
        -DT
        * v
        * np.cos(beta)
        / (l_r + l_f)
        * (1 / np.cos(steer_front) ** 2 + 1 / np.cos(steer_rear) ** 2)
    )

    return A, B, C


def linear_mpc_control(xref, xbar, x0, dref):
    """
    linear mpc control
    xref: reference point
    xbar: operational point
    x0: initial state
    dref: reference steer angle
    """

    x = cvxpy.Variable((NX, T + 1))
    u = cvxpy.Variable((NU, T))

    cost = 0.0
    constraints = []

    for t in range(T):
        cost += cvxpy.quad_form(u[:, t], R)

        if t != 0:
            cost += cvxpy.quad_form(xref[:, t] - x[:, t], Q)

        A, B, C = get_linear_model_matrix(
            xbar[2, t], xbar[3, t], dref[0, t], dref[1, t]
        )
        constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]

        if t < (T - 1):
            cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], Rd)
            constraints += [cvxpy.abs(u[1, t + 1] - u[1, t]) <= MAX_DSTEER * DT]
            constraints += [cvxpy.abs(u[2, t + 1] - u[2, t]) <= MAX_DSTEER * DT]

    cost += cvxpy.quad_form(xref[:, T] - x[:, T], Qf)

    constraints += [x[:, 0] == x0]
    constraints += [x[2, :] <= MAX_SPEED]
    constraints += [x[2, :] >= MIN_SPEED]
    constraints += [cvxpy.abs(u[0, :]) <= MAX_ACCEL]
    constraints += [cvxpy.abs(u[1, :]) <= MAX_STEER]
    constraints += [cvxpy.abs(u[2, :]) <= MAX_STEER]
    constraints += [cvxpy.abs(u[2, :]) <= 0.0001]

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    prob.solve(solver=cvxpy.ECOS, verbose=False, abstol=1e-7)

    if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
        ox = get_nparray_from_matrix(x.value[0, :])
        oy = get_nparray_from_matrix(x.value[1, :])
        ov = get_nparray_from_matrix(x.value[2, :])
        oyaw = get_nparray_from_matrix(x.value[3, :])
        oa = get_nparray_from_matrix(u.value[0, :])
        odelta_front = get_nparray_from_matrix(u.value[1, :])
        odelta_rear = get_nparray_from_matrix(u.value[2, :])

    else:
        print("Error: Cannot solve mpc..")
        oa, odelta_front, odelta_rear, ox, oy, oyaw, ov = (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    return oa, odelta_front, odelta_rear, ox, oy, oyaw, ov


def iterative_linear_mpc_control(xref, x0, dref, oa, od_f, od_r):
    """
    MPC contorl with updating operational point iteraitvely
    """

    if oa is None or od_f is None or od_r is None:
        oa = [0.0] * T
        od_f = [0.0] * T
        od_r = [0.0] * T

    for i in range(MAX_ITER):
        xbar = predict_motion(x0, oa, od_f, od_r, xref)
        poa, pod_f, pod_r = oa[:], od_f[:], od_r[:]  # previous oa and od
        oa, od_f, od_r, ox, oy, oyaw, ov = linear_mpc_control(xref, xbar, x0, dref)
        du = (
            sum(abs(oa - poa)) + sum(abs(od_f - pod_f)) + sum(abs(od_r - pod_r))
        )  # calc u change value
        if du <= DU_TH:
            break
    else:
        print("Iterative is max iter")

    return oa, od_f, od_r, ox, oy, oyaw, ov


def check_goal(state, goal, tind, nind):

    # check goal
    dx = state.x - goal[0]
    dy = state.y - goal[1]
    d = math.hypot(dx, dy)

    isgoal = d <= GOAL_DIS

    if abs(tind - nind) >= 5:
        isgoal = False

    isstop = abs(state.v) <= STOP_SPEED

    if isgoal and isstop:
        return True

    return False


if __name__ == "__main__":
    logger.info(f"type of R:{type(R)}")
    cx, cy, cyaw, ck, s = get_path_course()
    plt.plot(cx, cy, "-")
    # plt.show()

    state = State(x=0, y=0, yaw=cyaw[0], v=0.0)
    sp = calc_speed_profile(cx, cy, cyaw, TARGET_SPEED)
    # plot speed profile

    goal = [cx[-1], cy[-1]]
    logger.info(f"goal:{goal}")

    # initial yaw compensation
    if state.yaw - cyaw[0] >= math.pi:
        state.yaw -= math.pi * 2.0
    elif state.yaw - cyaw[0] <= -math.pi:
        state.yaw += math.pi * 2.0

    # initial state
    current_time = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    t = [0.0]
    df = [0.0]
    dr = [0.0]
    a = [0.0]
    target_ind, _ = calc_nearest_index(state, cx, cy, cyaw, 0)
    logger.info(f"target_ind:{target_ind}")
    odelta_f, odelta_r, oa = None, None, None

    cyaw = smooth_yaw(cyaw)
    dl = 1.0  # course tick
    while current_time < MAX_TIME:
        # get reference trajectory according to current velocity
        xref, target_ind, dref = calc_ref_trajectory(
            state, cx, cy, cyaw, ck, sp, dl, target_ind
        )
        print(f"dref:{dref}")
        x0 = [state.x, state.y, state.v, state.yaw]  # current state

        oa, odelta_f, odelta_r, ox, oy, oyaw, ov = iterative_linear_mpc_control(
            xref, x0, dref, oa, odelta_f, odelta_r
        )
        if odelta_f is not None and odelta_r is not None and oa is not None:
            dfi, dri, ai = odelta_f[0], odelta_r[0], oa[0]
        print(f"dfi:{dfi},dri:{dri},ai:{ai}")
        input("Press Enter to continue...")
        state = update_state(state, ai, dfi, dri)
        current_time += DT
        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(current_time)
        df.append(dfi)
        dr.append(dri)
        a.append(ai)
        if check_goal(state, goal, target_ind, len(cx)):
            print("Goal")
            break

    ## plot the result ,x,y, yaw
    down_sample = 2
    x = np.array(x)[::down_sample]
    y = np.array(y)[::down_sample]
    yaw = np.array(yaw)[::down_sample]
    plt.scatter(x, y, s=1)
    plt.quiver(x, y, np.cos(yaw), np.sin(yaw), scale=20)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()
