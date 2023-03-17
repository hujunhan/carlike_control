import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
# Define the transformation matrix
def transform_2d(x, y, theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, x],
                     [s, c, y],
                     [0, 0, 1]])
class Car:
    def __init__(self, x=0, y=0, theta=0.0, length=4, width=2):
        self.x = x
        self.y = y
        self.theta = theta
        self.length = length
        self.width = width
        self.w_v_transform = transform_2d(0, 0, 0)
        # Define the body points, four corners of the rectangle, 2d
        # defined by the length and width of the car
        self.body_points=np.asarray([[self.width/2, self.length/2 ,1],
                                     [self.width/2, -self.length/2,1],
                                     [-self.width/2, -self.length/2,1],
                                     [-self.width/2, self.length/2,1]]) # type: ignore
        self.body_points= np.dot(self.w_v_transform, self.body_points.T).T
    def draw(self):
        print(self.body_points)
        rect=Polygon(self.body_points[:,:2], closed=True, fill=False, edgecolor='r')
        
        self.w_v_transform=transform_2d(5,5,np.pi/6)
        self.body_points= np.dot(self.w_v_transform, self.body_points.T).T
        rect2=Polygon(self.body_points[:,:2], closed=True, fill=False, edgecolor='b')
        # Plot the car by draw the rectangle defined by the body points
        fig, ax = plt.subplots()
        ax.add_patch(rect)
        ax.add_patch(rect2)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_aspect('equal')
        # ax.autoscale()
        
        plt.show()
        
        

        

car = Car(x=0, y=0, theta=np.pi/4, length=4, width=2)
car.draw()
