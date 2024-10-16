import numpy as np
from scipy.spatial.transform import Rotation as R

panda_default_cartesian = ([3.06890567e-01, -1.31037208e-17,  5.90282052e-01,  9.23879533e-01, -3.82683432e-01, -4.85305078e-18,  6.82876066e-17])
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
