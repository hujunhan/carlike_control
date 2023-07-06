from carlike_control.car import Car
from carlike_control.controller import MPC
from carlike_control.viz import Visualization
import numpy as np
from carlike_control.path_planning import calc_spline_course
import math
import matplotlib.pyplot as plt
import rospy
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R

path = np.array([[0, 0], [0, 1.0],[-1,1.5],[-1,2]])
goal = path[-1]
trans_mat = np.asarray(
    [[0.75377273, 0.6571352, 0.0], [-0.6571352, 0.75377273, -0.0], [-0.0, 0.0, 1.0]]
)
INIT = True
REACH_GOAL = False

origin_position = [-2.65, -2.42]
yaw_offset = 0
ANIMATE = True
ROS = True
car = Car(x=0, y=0, yaw=np.pi / 2, v=0.0)


def odometry_callback(msg):
    """read the odometry message from the lidar SLAM and assign to car

    Args:
        msg (_type_): odometry message from the lidar SLAM
    """
    global INIT, origin_position, yaw_offset, REACH_GOAL
    ori = msg.pose.pose.orientation
    quant = [ori.x, ori.y, ori.z, ori.w]
    r = R.from_quat(quant)
    euler = r.as_euler("xyz")
    if INIT:
        origin_position = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        yaw_offset = np.pi / 2 - euler[2]
        INIT = False
        return
    new_p = [
        msg.pose.pose.position.x - origin_position[0],
        msg.pose.pose.position.y - origin_position[1],
        1,
    ]

    new_p = np.matmul(trans_mat, new_p)
    # check if the car reaches the goal
    if np.linalg.norm(new_p[:2] - goal) < 0.05:
        REACH_GOAL = True
    car.x = new_p[0]
    car.y = new_p[1]
    car.yaw = euler[2] + yaw_offset
    print(f"current pose: {car.x:.2f}, {car.y:.2f}, {car.yaw/np.pi*180:.2f}")


def motor_callback(msg):
    global motor_state
    global car
    motor_state = msg.data[
        1:9
    ]  # 0-3 motor speed, m/s, 4-7 motor steer, rad(-pi-pi, anti-clockwise), order: front left, rear left, front right, rear right
    ## update the car state
    car.steer_front = (motor_state[4] + motor_state[6]) / 2
    car.steer_rear = (motor_state[5] + motor_state[7]) / 2
    temp_speed=(motor_state[0] + motor_state[1] + motor_state[2] + motor_state[3]) / 4
    if temp_speed>0.01:
        car.v = (motor_state[0] + motor_state[1] + motor_state[2] + motor_state[3]) / 4

    print(f"motor:{motor_state}")


if ROS:
    rospy.init_node("carlike_control")
    pub = rospy.Publisher("AgvControl", Float32MultiArray, queue_size=10)
    motor_sub = rospy.Subscriber(
        "AgvData", Float32MultiArray, queue_size=10, callback=motor_callback
    )
    odom_sub = rospy.Subscriber(
        "odom", Odometry, queue_size=10, callback=odometry_callback
    )
    r = rospy.Rate(10)
## Parameters
TARGET_SPEED = 1.0
MAX_TIME = 500.0
N_IND_SEARCH = 50  # Search index number
DT = 0.1  # time tick [s]
# Path definition
T = 5


def pi_2_pi(angle):
    return np.mod(angle + np.pi, 2 * np.pi) - np.pi


def get_path_course(path=path):
    x = path[:, 0]
    y = path[:, 1]
    ds = 0.1
    rx, ry, ryaw, rk, s = calc_spline_course(x, y, ds)
    return rx, ry, ryaw, rk, s


def calc_speed_profile(cx, cy, cyaw, ck, target_speed):
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
    ## calculate the speed profile based on the curvature
    # the speed should be lower when the curvature is higher
    for i in range(0, len(cx)):
        if abs(ck[i]) > 1.0:
            speed_profile[i] /= abs(ck[i])
    speed_profile[-1] = 0.0

    return speed_profile


def calc_nearest_index(state: Car, cx, cy, cyaw, pind):
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


def calc_ref_trajectory(state: Car, cx, cy, cyaw, ck, sp, dl, pind):
    """get the reference trajectory, the next few points in the path,
    related to the current speed, get the reference path properties like
    x,y,speed,yaw

    """
    xref = np.zeros((4, T + 1))
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


if __name__ == "__main__":
    ## Visualization
    from carlike_control.viz import Visualization
    from carlike_control.car import Car
    import time

    if ANIMATE:
        viz = Visualization((-3, 3), (-3, 3))
        plt.show(block=False)
    ## Path
    cx, cy, cyaw, ck, s = get_path_course()
    sp = calc_speed_profile(cx, cy, cyaw, ck, TARGET_SPEED)

    ## Car
    if car.yaw - cyaw[0] >= math.pi:
        car.yaw -= math.pi * 2.0
    elif car.yaw - cyaw[0] <= -math.pi:
        car.yaw += math.pi * 2.0
    ## MPC
    controller = MPC(car)
    goal = [cx[-1], cy[-1]]

    # initial state
    current_time = 0.0
    target_ind, _ = calc_nearest_index(car, cx, cy, cyaw, 0)
    import time

    record_state = []
    cyaw = smooth_yaw(cyaw)
    dl = 1.0  # course tick
    steer_history = []
    try:
        while current_time < MAX_TIME and not REACH_GOAL:
            # get reference trajectory according to current velocity
            start = time.time()
            xref2, target_ind, dref = calc_ref_trajectory(
                car, cx, cy, cyaw, ck, sp, dl, target_ind
            )

            controller.solve(car, xref2)
            u = controller.u_perterb.value[:, 0]
            end = time.time()
            # print(f"elapsed time:{end-start}")
            temp_v = car.update_state(u[0], u[1], u[2])
            # car.update_all_steer([u[1], u[1], u[2], u[2]])
            a, b, c, d = car.update_all_steer_simple(u[1])
            current_time += DT
            record_state.append(
                [car.x, car.y, car.v, car.yaw, car.steer_front, car.steer_rear]
            )
            steer_history.append([car.steer_front, car.steer_rear])
            if ANIMATE:
                viz.clear()
                viz.draw_car(car)
                viz.draw_path(cx, cy)
                viz.ax.set_title(f"speed:{car.v:.2f}m/s")
                plt.pause(0.1)
            if ROS:
                msg = Float32MultiArray()
                msg.data = [1, temp_v, temp_v, temp_v, temp_v, a, c, b, d]
                print(msg.data)
                pub.publish(msg)
                r.sleep()
    except:
        msg = Float32MultiArray()
        msg.data = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        print(msg.data)
        pub.publish(msg)
    msg = Float32MultiArray()
    msg.data = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    print(msg.data)
    pub.publish(msg)
    # plot the steer history
    # steer_history = np.array(steer_history)
    # plt.plot(range(len(steer_history)), steer_history[:, 0], label="front")
    # plt.plot(range(len(steer_history)), steer_history[:, 1], label="rear")
    # plt.legend()
    # plt.show()
