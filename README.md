# carlike_control
Control for car-like mobile robot

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


## References

[1] https://ftp.idu.ac.id/wp-content/uploads/ebook/tdg/TERRAMECHANICS%20AND%20MOBILITY/epdf.pub_vehicle-dynamics-and-control-2nd-edition.pdf