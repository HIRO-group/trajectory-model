from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

from read import read_vectors

# file_name = '01-09-2023 13-42-14'
# file_name = '01-09-2023 13-58-43'
file_name = "01-09-2023 14-09-56"

file_path = '/home/ava/projects/assets/cartesian/'+file_name+'/cartesian_positions.bin'
vectors = read_vectors(file_path)
rolls, pitches, yaws = [], [], []
for pos in vectors:
    a, b, c, d = pos[3], pos[4], pos[5], pos[6]
    rotation_matrix = R.from_quat([a, b, c, d])
    euler = rotation_matrix.as_euler('xyz', degrees=True)
    roll = euler[0]
    pitch = euler[1]
    yaw = euler[2]
    rolls.append(roll)
    pitches.append(pitch)
    yaws.append(yaw)


plt.subplot(3, 1, 1)
plt.plot(rolls)
plt.ylabel('Roll')

plt.subplot(3, 1, 2) 
plt.plot(pitches)
plt.ylabel('Pitch')

plt.subplot(3, 1, 3)
plt.plot(yaws) 
plt.ylabel('Yaw')

plt.xlabel('Sample')
plt.tight_layout()


# plt.savefig('plots/angles/{file_name}.png'.format(file_name=file_name), dpi=300)
plt.show()