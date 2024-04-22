import numpy as np
from scipy.spatial.transform import Rotation as R


panda_a, panda_b, panda_c, panda_w = 0.92861403, -0.37020025, -0.0051489, 0.02452055
cup_a, cup_b, cup_c, cup_w = 0, 0, 0, 1
cup_q  = R.from_quat([cup_a, cup_b, cup_c, cup_w]).as_matrix()
panda_q_inv = R.from_quat([-panda_a, -panda_b, -panda_c, panda_w]).as_matrix()
diff = np.dot(cup_q, panda_q_inv)


def rotate_ee_matrix(panda_ee_matrix):
    new_panda_mat = np.dot(diff, panda_ee_matrix)
    return new_panda_mat


def rotate_panda_traj(panda_cartesian_traj):
    for cartesian in panda_cartesian_traj:
        ee_matrix = R.from_quat(cartesian[3:]).as_matrix()
        new_ee_matrix = rotate_ee_matrix(ee_matrix)
        new_quat = R.from_matrix(new_ee_matrix).as_quat()
        cartesian[3:] = new_quat
    return panda_cartesian_traj