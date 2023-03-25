import cvxpy
import numpy as np
from carlike_control.car import Car
import copy
import math


class MPC:
    def __init__(self, car: Car) -> None:
        ## car model
        self.sim_car = car

        self.DT = 0.1
        NX = 4
        NU = 3
        T = 5
        self.NX = NX
        self.NU = NU
        self.T = T
        ## cvxpy variables and parameters
        # cost matrix
        self.U_cost = np.diag([0.01, 0.01, 0.01])
        self.U_diff_cost = np.diag([0.01, 2.0, 2.0])
        self.state_cost = np.diag([1.0, 1.0, 0.5, 0.5])
        self.state_diff_cost = np.diag([1.0, 1.0, 0.5, 0.5])
        # parameters, input of the problem
        self.A = [cvxpy.Parameter((NX, NX)) for i in range(T)]
        self.B = [cvxpy.Parameter((NX, NU)) for i in range(T)]
        self.C = [cvxpy.Parameter(NX) for i in range(T)]
        self.xref = cvxpy.Parameter((NX, T + 1))
        self.current_x = cvxpy.Parameter(NX)
        # variables, output of the problem
        self.x = cvxpy.Variable((NX, T + 1))
        self.u = cvxpy.Variable((NU, T))
        self.u_perterb = cvxpy.Variable((NU, T))
        self.problem: cvxpy.Problem = self.get_optimization_problem()

        ## last time solution
        self.last_u = np.zeros((NU, T))

    def solve(self, car: Car, xref):
        """solve the optimization problem

        Args:
            car (Car): control object
            xref (_type_): reference state

        Returns:
            _type_: _description_
        """
        ## update parameters
        self.current_x.value = [car.x, car.y, car.v, car.yaw]
        self.xref.value = xref
        xpred = self.predict_x(car, self.last_u)

        for t in range(self.T):
            (
                self.A[t].value,
                self.B[t].value,
                self.C[t].value,
            ) = self.get_linear_model_matrix(xpred[2, t], xpred[3, t], 0, 0)
        self.problem.solve(solver=cvxpy.SCS, verbose=False, max_iters=1000)
        if (
            self.problem.status == cvxpy.OPTIMAL
            or self.problem.status == cvxpy.OPTIMAL_INACCURATE
        ):
            self.last_u = self.u_perterb.value
            return self.u.value

    def predict_x(self, car: Car, u):
        """predict the state of the car according to the input (last time solution)

        Args:
            car (Car): _description_
            u (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = np.zeros((self.NX, self.T + 1))
        x[:, 0] = [car.x, car.y, car.v, car.yaw]
        temp_car = copy.deepcopy(car)
        for i in range(self.T):
            temp_car.update_state(u[0, i], u[1, i], u[2, i])
            x[:, i + 1] = [temp_car.x, temp_car.y, temp_car.v, temp_car.yaw]
        return x

    def get_linear_model_matrix(
        self, v: float, yaw: float, steer_front: float, steer_rear: float, DT=0.1
    ):
        """linear model, calculated by first order Taylor expansion

        Args:
            v (float): _description_
            yaw (float): _description_
            steer_front (float): _description_
            steer_rear (float): _description_
            DT (float, optional): _description_. Defaults to 0.1.

        Returns:
            _type_: _description_
        """
        beta = self.calc_beta(steer_front, steer_rear)
        A = np.identity(4)
        A[0, 2] = DT * math.cos(yaw + beta)
        A[0, 3] = -DT * v * math.sin(yaw + beta)
        A[1, 2] = DT * math.sin(yaw + beta)
        A[1, 3] = DT * v * math.cos(yaw + beta)
        A[3, 2] = (
            DT
            * np.cos(beta)
            * (np.tan(steer_front) - np.tan(steer_rear))
            / (self.sim_car.l_r + self.sim_car.l_f)
        )

        B = np.zeros((4, 3))
        B[2, 0] = DT
        B[3, 1] = (
            DT
            * v
            * np.cos(beta)
            / (self.sim_car.l_r + self.sim_car.l_f)
            / np.cos(steer_front) ** 2
        )
        B[3, 2] = (
            -DT
            * v
            * np.cos(beta)
            / (self.sim_car.l_r + self.sim_car.l_f)
            / np.cos(steer_rear) ** 2
        )

        C = np.zeros(4)
        C[0] = DT * v * math.sin(yaw + beta) * yaw
        C[1] = -DT * v * math.cos(yaw + beta) * yaw
        C[3] = (
            -DT
            * v
            * np.cos(beta)
            / (self.sim_car.l_r + self.sim_car.l_f)
            * (1 / np.cos(steer_front) ** 2 - 1 / np.cos(steer_rear) ** 2)
        )
        return A, B, C

    def calc_beta(self, steer_front, steer_rear):
        return math.atan2(
            (
                self.sim_car.l_f * np.tan(steer_rear)
                + self.sim_car.l_r * np.tan(steer_front)
            ),
            (self.sim_car.l_f + self.sim_car.l_r),
        )

    def get_optimization_problem(self):
        """formulate the optimization problem
        objective: minimize the command, state error and input change and state change
        constraints: speed limit, steering limit, acceleration limit, change rate limit

        Returns:
            _type_: _description_
        """
        cost = 0.0
        constraints = []
        xref = self.xref
        u = self.u
        u_perterb = self.u_perterb
        current_x = self.current_x
        T = self.T
        x = self.x
        A = self.A
        B = self.B
        C = self.C
        for t in range(self.T):
            cost += cvxpy.quad_form(u[:, t], self.U_cost)

            if t != 0:
                cost += cvxpy.quad_form(xref[:, t] - x[:, t], self.state_cost)

            constraints += [x[:, t + 1] == A[t] @ x[:, t] + B[t] @ u[:, t] + C[t]]

            if t < (T - 1):
                cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], self.U_diff_cost)
                constraints += [
                    cvxpy.abs(u[1, t + 1] - u[1, t])
                    <= self.sim_car.MAX_DSTEER * self.DT
                ]
                constraints += [
                    cvxpy.abs(u[2, t + 1] - u[2, t])
                    <= self.sim_car.MAX_DSTEER * self.DT
                ]

        cost += cvxpy.quad_form(xref[:, T] - x[:, T], self.state_diff_cost)

        constraints += [x[:, 0] == current_x]
        constraints += [x[2, :] <= self.sim_car.MAX_SPEED]
        constraints += [x[2, :] >= self.sim_car.MIN_SPEED]
        constraints += [cvxpy.abs(u[0, :]) <= self.sim_car.MAX_ACCEL]
        constraints += [cvxpy.abs(u[1, :]) <= self.sim_car.MAX_STEER]
        constraints += [cvxpy.abs(u[2, :]) <= self.sim_car.MAX_STEER]
        constraints += [u_perterb == u + 1e-3 * np.random.randn(self.NU, T)]
        prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        return prob
