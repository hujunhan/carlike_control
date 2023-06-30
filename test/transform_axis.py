import scipy
import numpy as np
from scipy.spatial.transform import Rotation as R

dx = 0.78
dy = 0.68

# normalize dx, dy
dx, dy = dx / np.sqrt(dx**2 + dy**2), dy / np.sqrt(dx**2 + dy**2)
p = [dx, dy, 1]
print(dx, dy)
theta = np.arctan2(dy, dx)

# calculate rotation matrix from theta
r = R.from_euler("z", -theta)
trans_mat = r.as_matrix()
print(trans_mat)

# calculate the new point by multiplying the rotation matrix
new_p = np.matmul(trans_mat, p)
print(new_p)
