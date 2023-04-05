## Script to create the environment for the carlike robot
# include Obstacle class and Environment class
import numpy as np
from typing import List
import math


class Obstacle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        ## velocity of the obstacle, vector
        self.v = np.array([0, 0])
        self.shape = None

    def rectangle(self, width, height):
        self.width = width
        self.height = height
        self.shape = "rectangle"

    def circle(self, radius):
        self.radius = radius
        self.shape = "circle"

    def update(self, dt):
        self.x += self.v[0] * dt
        self.y += self.v[1] * dt


class Environment:
    def __init__(self, width, height, num_obstacles):
        self.width = width
        self.height = height
        self.obstacles: List[Obstacle] = []
        self.delta = 0.5  # grace period for collision detection
        # Add random obstacles
        for i in range(num_obstacles):
            x = np.random.uniform(0, self.width)
            y = np.random.uniform(0, self.height)
            shape = np.random.choice(["rectangle", "circle"])
            self.add_obstacle(x, y, shape)

    def add_obstacle(self, x, y, shape):
        obstacle = Obstacle(x, y)

        if shape == "rectangle":
            # add random width and height rectangle
            obstacle.rectangle(np.random.uniform(2, 5), np.random.uniform(2, 5))
        elif shape == "circle":
            # add random radius circle
            obstacle.circle(np.random.uniform(2, 5))
        self.obstacles.append(obstacle)

    def is_in_obstacle(self, x, y):
        delta = self.delta
        # check if the point still in the boundary
        if x < 0 or x > self.width or y < 0 or y > self.height:
            return True
        for obstacle in self.obstacles:
            if obstacle.shape == "rectangle":
                if (
                    x >= obstacle.x - obstacle.width / 2 - delta
                    and x <= obstacle.x + obstacle.width / 2 + delta
                    and y >= obstacle.y - obstacle.height / 2 - delta
                    and y <= obstacle.y + obstacle.height / 2 + delta
                ):
                    return True
            elif obstacle.shape == "circle":
                if (
                    np.sqrt((x - obstacle.x) ** 2 + (y - obstacle.y) ** 2)
                    <= obstacle.radius + delta
                ):
                    return True
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
        delta = self.delta
        # check if the point in the obstacle
        if self.is_in_obstacle(x1, y1) or self.is_in_obstacle(x2, y2):
            return True

        for obstacle in self.obstacles:
            if obstacle.shape == "rectangle":
                # check four sides of the rectangle
                # use self._intersect to check if the line cross the side
                # given two points of the side and two points of the line
                # fmt: off
                for i in range(4):
                    if i == 0:
                        A = (obstacle.x - obstacle.width / 2, obstacle.y - obstacle.height / 2)
                        B = (obstacle.x + obstacle.width / 2, obstacle.y - obstacle.height / 2)
                    elif i == 1:
                        A = (obstacle.x + obstacle.width / 2, obstacle.y - obstacle.height / 2)
                        B = (obstacle.x + obstacle.width / 2, obstacle.y + obstacle.height / 2)
                    elif i == 2:
                        A = (obstacle.x + obstacle.width / 2, obstacle.y + obstacle.height / 2)
                        B = (obstacle.x - obstacle.width / 2, obstacle.y + obstacle.height / 2)
                    elif i == 3:
                        A = (obstacle.x - obstacle.width / 2, obstacle.y + obstacle.height / 2)
                        B = (obstacle.x - obstacle.width / 2, obstacle.y - obstacle.height / 2)
                    if self._intersect(A, B, (x1, y1), (x2, y2)):
                        return True
                # fmt: on
            if obstacle.shape == "circle":
                if self._line_circle_intersection(
                    (x1, y1), (x2, y2), (obstacle.x, obstacle.y), obstacle.radius
                ):
                    return True
        return False

    def _ccw(self, A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    # Return true if line segments AB and CD intersect
    def _intersect(self, A, B, C, D):
        return self._ccw(A, C, D) != self._ccw(B, C, D) and self._ccw(
            A, B, C
        ) != self._ccw(A, B, D)

    def _line_circle_intersection(self, start, end, center, radius):
        """Check if a line intersects a circle.

        Args:
            start (tuple): Starting point of the line, e.g. (x1, y1)
            end (tuple): Ending point of the line, e.g. (x2, y2)
            center (tuple): Center point of the circle, e.g. (xc, yc)
            radius (float): Radius of the circle

        Returns:
            bool: True if the line intersects the circle, False otherwise
        """
        # Calculate the direction of the line
        dx = end[0] - start[0]
        dy = end[1] - start[1]

        # Calculate the distance between the line and the center of the circle
        a = dx**2 + dy**2
        b = 2 * (dx * (start[0] - center[0]) + dy * (start[1] - center[1]))
        c = (
            center[0] ** 2
            + center[1] ** 2
            + start[0] ** 2
            + start[1] ** 2
            - 2 * (center[0] * start[0] + center[1] * start[1])
            - radius**2
        )

        # Check if the discriminant is less than zero, which means no intersection
        discriminant = b**2 - 4 * a * c
        if discriminant < 0:
            return False

        # Calculate the two possible solutions for t (where the line intersects the circle)
        t1 = (-b + math.sqrt(discriminant)) / (2 * a)
        t2 = (-b - math.sqrt(discriminant)) / (2 * a)

        # Check if the intersection point is between the start and end points of the line
        if 0 <= t1 <= 1 or 0 <= t2 <= 1:
            return True

        return False
