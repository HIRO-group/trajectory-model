import matplotlib.pyplot as plt
from ee_vel import get_ee_vel

# file_name = '01-09-2023 13-42-14'
# file_name = '01-09-2023 13-58-43'
file_name = "01-09-2023 14-09-56"

x_vels, y_vels, z_vels = get_ee_vel(file_name)

dt = 0.001  
x_pos = 0
x_accel = []
x_pos_samples = []

for i in range(1, len(x_vels)):
    x_accel.append((x_vels[i] - x_vels[i-1])/ dt)


y_pos = 0
y_accel = []
y_pos_samples = []

for i in range(1, len(y_vels)):
    y_accel.append((y_vels[i] - y_vels[i-1])/ dt)


z_pos = 0
z_accel = []
z_pos_samples = []

for i in range(1, len(z_vels)):
    z_accel.append((z_vels[i] - z_vels[i-1])/ dt)


plt.subplot(3, 1, 1)
plt.plot(x_accel)
plt.ylabel('X')

plt.subplot(3, 1, 2)
plt.plot(y_accel)
plt.ylabel('Y')

plt.subplot(3, 1, 3)
plt.plot(z_accel)
plt.ylabel('Z')

plt.xlabel('Sample')
plt.tight_layout()
plt.savefig('/home/ava/projects/trajectory-model/plots/accel/{file_name}.png'.format(file_name=file_name), dpi=300)
plt.show()
