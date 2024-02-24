import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from trajectory_model.helper.helper import rotate_vector, calculate_endpoint, quat_to_euler

panda_default_cartesian = np.array([0.306891, -2.36797e-16, 0.590282, 0.707107, -0.707107, -6.12323e-17, -0.707107])
cup_default_cartesian = np.array([-0.007746216841042042,0.04685171693563461,0.16710145771503448,-0.0044183372519910336,-0.0012599572073668242,-0.21964260935783386,-0.9755696058273315])

panda_a, panda_b, panda_c, panda_w = panda_default_cartesian[3], panda_default_cartesian[4], panda_default_cartesian[5], panda_default_cartesian[6]
cup_a, cup_b, cup_c, cup_w = cup_default_cartesian[3], cup_default_cartesian[4], cup_default_cartesian[5], cup_default_cartesian[6]


panda_q = R.from_quat([panda_a, panda_b, panda_c, panda_w]).as_matrix()
cup_q  = R.from_quat([cup_a, cup_b, cup_c, cup_w]).as_matrix()

panda_q_inv = R.from_quat([-panda_a, -panda_b, -panda_c, panda_w]).as_matrix()
diff = np.dot(cup_q, panda_q_inv)


x, y, z = 0, 0, 0

# panda = q1
# cup = q2

# diff * panda = cup  --->  diff = cup * inverse(panda)
# where:  inverse(panda) = conjugate(panda) / abs(panda)
# and:  conjugate( quaternion(re, i, j, k) ) = quaternion(re, -i, -j, -k)


new_panda_mat = np.dot(diff, panda_q)
new_panda_quat = R.from_matrix(new_panda_mat).as_quat()

# print(new_panda_quat)

panda_a, panda_b, panda_c, panda_w = new_panda_quat[0], new_panda_quat[1], new_panda_quat[2], new_panda_quat[3]

start_point = np.array([[x, y, z]])
panda_end_point = calculate_endpoint(start_point, panda_a, panda_b, panda_c, panda_w)
cup_end_point = calculate_endpoint(start_point, cup_a, cup_b, cup_c, cup_w)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.quiver(0, 0, 0, panda_end_point[0][0], panda_end_point[0][1], panda_end_point[0][2], length = 0.1, normalize = True, color='red', label='panda')
ax.quiver(1, 0, 0, cup_end_point[0][0], cup_end_point[0][1], cup_end_point[0][2], length = 0.1, normalize = True, color='blue', label='cup')


ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.legend()
plt.show()