import numpy as np
from constants import Tasks
from trajectory_model.helper.read import read_panda_vectors
from trajectory_model.helper.helper import quat_to_euler
from trajectory_model.helper.rotate_quaternion import rotate_panda_to_match_orientation

def get_ee_range():
    x, y, z, phi, theta, psi = [], [], [], [], [], []
    for task in Tasks.all:
        for file_name in task:
            cartesian_p = '/home/ava/projects/assets/cartesian/'+file_name+'/cartesian_positions.bin'
            trajectory = read_panda_vectors(cartesian_p)
            panda_trajectory = np.array([np.array(trajectory[i]) for i in range(0, len(trajectory))])
            vectors = rotate_panda_to_match_orientation(panda_trajectory)
            for vector in vectors:
                x.append(vector[0])
                y.append(vector[1])
                z.append(vector[2])
                a, b, c, d = vector[3], vector[4], vector[5], vector[6]
                euler_angles = quat_to_euler([a, b, c, d])
                phi.append(euler_angles[0])
                theta.append(euler_angles[1])
                psi.append(euler_angles[2])
    
    return x, y, z, phi, theta, psi
    

if __name__ == "__main__":
    x, y, z, phi, theta, psi = get_ee_range()
    print("x: ", np.min(x), np.max(x))
    print("y: ", np.min(y), np.max(y))
    print("z: ", np.min(z), np.max(z))
    print("phi: ", np.min(phi), np.max(phi))
    print("theta: ", np.min(theta), np.max(theta))
    print("psi: ", np.min(psi), np.max(psi))

# rotated panda:
# x:  -0.08586814095362721 0.8194352670560375
# y:  -0.6626778097186472 0.663390310924573
# z:  0.20620225301190295 0.7921202881391447
# phi:  -179.87093416628917 179.9494111795403
# theta:  -89.40684169320777 19.932033221712146
# psi:  -179.91577607007912 179.91950224711127