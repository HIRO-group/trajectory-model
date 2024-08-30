import math

def rotate_quaternion(q, angle, axis):
    x, y, z, w = q
    sin_a = math.sin(angle/2)
    cos_a = math.cos(angle/2)
    x_axis = (x, y, z)
    x_prime = x * cos_a + (x_axis[axis] * sin_a) 
    y_prime = y * cos_a + (x_axis[(axis+1)%3] * sin_a)
    z_prime = z * cos_a + (x_axis[(axis+2)%3] * sin_a)  
    w_prime = w * cos_a - (x_axis[axis] * sin_a)
    return x_prime, y_prime, z_prime, w_prime


def quaternion_to_angle_axis(q1, q2):
    x1, y1, z1, w1 = q1 
    x2, y2, z2, w2 = q2
    
    cos_a = w1*w2 + x1*x2 + y1*y2 + z1*z2
    if cos_a < 0:
        w2 = -w2
        x2 = -x2
        y2 = -y2
        z2 = -z2
        cos_a = -cos_a
        
    angle = math.acos(cos_a) * 2
    
    if abs(angle) < 0.00001:
        return 0, 0
    
    sin_a = math.sqrt(1 - cos_a*cos_a)
    
    x = (y1*z2 - z1*y2)/sin_a 
    y = (z1*x2 - x1*z2)/sin_a
    z = (x1*y2 - y1*x2)/sin_a
    
    axis = 0 if abs(x) > abs(y) and abs(x) > abs(z) else \
           1 if abs(y) > abs(x) and abs(y) > abs(z) else 2
           
    return angle, axis

def rotate_quaternion_list(quaternions, q):
    q_start = quaternions[0] 
    angle, axis = quaternion_to_angle_axis(q, q_start)
    
    rotated = []
    for x, y, z, w in quaternions:
        x_rot, y_rot, z_rot, w_rot = rotate_quaternion((x, y, z, w), angle, axis)
        rotated.append((x_rot, y_rot, z_rot, w_rot))
    return rotated


import numpy as np
from scipy.spatial.transform import Rotation as R

panda_default_cartesian = np.array([0.306891, -2.36797e-16, 0.590282, 0.707107, -0.707107, -6.12323e-17, -0.707107])
cup_default_cartesian = np.array([-0.007746216841042042,0.04685171693563461,0.16710145771503448,-0.0044183372519910336,-0.0012599572073668242,-0.21964260935783386,-0.9755696058273315])
panda_a, panda_b, panda_c, panda_w = panda_default_cartesian[3], panda_default_cartesian[4], panda_default_cartesian[5], panda_default_cartesian[6]
cup_a, cup_b, cup_c, cup_w = cup_default_cartesian[3], cup_default_cartesian[4], cup_default_cartesian[5], cup_default_cartesian[6]
panda_q = R.from_quat([panda_a, panda_b, panda_c, panda_w]).as_matrix()
cup_q  = R.from_quat([cup_a, cup_b, cup_c, cup_w]).as_matrix()
panda_q_inv = R.from_quat([-panda_a, -panda_b, -panda_c, panda_w]).as_matrix()
diff = np.dot(cup_q, panda_q_inv)


def rotate_panda_to_match_orientation(panda_trajectory):
    # diff * panda = cup  --->  diff = cup * inverse(panda)
    # where:  inverse(panda) = conjugate(panda) / abs(panda)
    # and:  conjugate( quaternion(re, i, j, k) ) = quaternion(re, -i, -j, -k)
    for step in range(panda_trajectory.shape[0]):
        panda_q = [panda_trajectory[step][3], panda_trajectory[step][4], panda_trajectory[step][5], panda_trajectory[step][6]]
        panda_q = R.from_quat(panda_q).as_matrix()
        new_panda_mat = np.dot(diff, panda_q)
        new_panda_quat = R.from_matrix(new_panda_mat).as_quat()
        panda_a, panda_b, panda_c, panda_w = new_panda_quat[0], new_panda_quat[1], new_panda_quat[2], new_panda_quat[3]
        panda_trajectory[step][3], panda_trajectory[step][4], panda_trajectory[step][5], panda_trajectory[step][6] = panda_a, panda_b, panda_c, panda_w
    return panda_trajectory
