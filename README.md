# carlike_control
Control for car-like mobile robot

## Usage

```bash
# Terminal 1
roslaunch hdl_localization indoor.launch

# Terminal 2
roscd hdl_localization/rviz
rviz -d hdl_localization.rviz

## Terminal 3
roslaunch lslidar_driver lslidar_c16.launch

## Terminal 4
rosservice call /relocalize

## Terminal 5
rosrun HubMotor_pkg HubMotor

## Run the control script
python3 ./test/FWS_ros.py
```

## Overview

Control the car-like mobile robot

TODO:

- [x] Add a description of the package
- [x] Add Wheel class
  - width
  - radius
- [x] More on Car class
  - [x] steer limit
  - [x] add wheels
- [x] Add a Visualization class
  - [x] Add a method to draw a car with wheels
  - [x] Add a method to draw a path
- [x] Add dynamic simulation
  - [x] Add a method to simulate the car
  - [ ] Understand how to calculate the car's velocity

## Changelog

* add the car class, show in world frame
* refactor the code, add the wheel class, the viz class and test case for each class
* Add the dynamic simulation using bicycle model described in [1] page 20
* Add path planning (cubic spline interpolation)
* Add path tracking (MPC) // 2023.3.21
* Format the code
* Add MPC control for 4ws car (problem, need fix) // 2023.3.22
* Fix the problem of MPC control for 4ws car (wrong B, C matrix) 
* Add visualization for MPC control for 4ws car // 2023.3.23
* Speed up code for MPC control for 4ws car, by using cvxpy's parameter feature // 2023.3.24
* Add MPC controller class, add MPC control for 4ws car // 2023.3.24
* Add Environment class, prepare for the simulation of the car in the environment // 2023.4.4
* Add Environment collision detection, and BIT path planning (need fix) // 2023.4.5
* Fix BIT (enlarge the sample size) // 2023.4.5
* Update four wheels steer according to bycicle model // 2023.4.10
* replanning after obstacles changes // 2023.4.11
* Test with real robot (Only with simulation state update) //2023.6.6
* Test with real robot (with MPC control and lidar/encoder Feedback) //2023.7.5
* Add via points plot, goal check, accel/velocity error fix //2023.7.5
* Update test instructions in readme //2023.7.5
* add velocity constrain at destination //2023.7.6


## References

[1] https://ftp.idu.ac.id/wp-content/uploads/ebook/tdg/TERRAMECHANICS%20AND%20MOBILITY/epdf.pub_vehicle-dynamics-and-control-2nd-edition.pdf