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