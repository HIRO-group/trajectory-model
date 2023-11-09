from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

from trajectory_model.helper.read import read_panda_vectors

def get_roll_pitch_yaw(file_name):
    file_path = '/home/ava/projects/assets/cartesian/'+file_name+'/cartesian_positions.bin'
    vectors = read_panda_vectors(file_path)
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
    return rolls, pitches, yaws

small_full_file_names = ["10-09-2023 13-14-07", "10-09-2023 13-10-26", "10-09-2023 10-03-18",
                    "10-09-2023 13-12-16", "10-09-2023 12-36-43"]
small_half_file_names = ["10-09-2023 13-30-09", "10-09-2023 13-32-29", "10-09-2023 13-39-37"]

big_half_file_names = ["01-09-2023 13-42-14", "01-09-2023 13-58-43", "01-09-2023 14-09-56"]

big_full_file_names = ["10-09-2023 10-06-37", "10-09-2023 12-25-04", "10-09-2023 12-29-22"]


file_names = [small_full_file_names[0], small_half_file_names[0], 
              big_full_file_names[0], big_half_file_names[0]]

experiment_names = ["Champagne 0.8", "Champagne 0.5", "Wine 0.8", "Wine 0.3"]

roll_data, pitch_data, yaw_data = [], [], []

for file_name in file_names:
    rolls, pitches, yaws = get_roll_pitch_yaw(file_name)
    roll_data.append(rolls)
    pitch_data.append(pitches)
    yaw_data.append(yaws)


fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

for i in range(4):
    axs[0].plot(roll_data[i], label=experiment_names[i])

for i in range(4):
    axs[1].plot(pitch_data[i], label=experiment_names[i])

for i in range(4):
    axs[2].plot(yaw_data[i], label=experiment_names[i])

for ax in axs:
    ax.legend()

axs[0].set_ylabel("Roll")
axs[1].set_ylabel("Pitch")
axs[2].set_ylabel("Yaw")
axs[2].set_xlabel("Time")

plt.suptitle("Roll, Pitch, and Yaw Data from 4 Experiments")
plt.savefig(f'plots/angles/{file_names}.png'.format(file_names=','.join(file_names)), dpi=300)
plt.show()


# plt.subplot(3, 1, 1)
# plt.plot(rolls)
# plt.ylabel('Roll')

# plt.subplot(3, 1, 2) 
# plt.plot(pitches)
# plt.ylabel('Pitch')

# plt.subplot(3, 1, 3)
# plt.plot(yaws) 
# plt.ylabel('Yaw')

# plt.xlabel('Sample')
# plt.tight_layout()


# plt.savefig('plots/angles/{file_name}.png'.format(file_name=file_name), dpi=300)
# plt.show()