from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import numpy as np

from trajectory_model.helper.read import read_panda_vectors
from trajectory_model.helper.jacobian import get_jacobian

from sandbox.constants import Tasks


def get_time_taken(file_name):
    file_path_v = '/home/ava/projects/assets/'+file_name+'/joint_velocities.bin'
    joint_velocities = read_panda_vectors(file_path_v)
    time = len(joint_velocities)/1000
    return time

if __name__ == "__main__":
    time_task_mean, time_task_median, time_task_max = [], [], []
    for task in [Tasks.task_1, Tasks.task_2, Tasks.task_3, Tasks.task_4]:
        times = []
        for file_name in task:
            times.append(get_time_taken(file_name))
        time_task_mean.append(np.mean(times))
        time_task_median.append(np.median(times))
        time_task_max.append(np.max(times))
        
    print('Mean: ', time_task_mean)
    print('Median: ', time_task_median)
    print('Max: ', time_task_max)
    # raise Exception
    fig = plt.figure()
    bar_width = 0.4
    br1 = np.arange(2)
    br2 = [x + bar_width for x in br1]

    plt.bar(br1, [time_task_mean[0], time_task_mean[2]], color ='pink', width = bar_width, edgecolor ='pink')
    plt.bar(br2, [time_task_mean[1], time_task_mean[3]], color ='orange', width = bar_width, edgecolor ='orange')

    
    # plt.text(br1[0], task_4_mean/4, "Fill-Level: 30%", ha='center', va='bottom')
    # plt.text(br1[1], task_4_mean/4, "Fill-Level: 50%", ha='center', va='bottom')
    # plt.text(br1[2], task_4_mean/4, "Fill-Level: 30%", ha='center', va='bottom')
    # plt.text(br2[0], task_4_mean/4, "Fill-Level: 80%", ha='center', va='bottom')
    # plt.text(br2[1], task_4_mean/4, "Fill-Level: 80%", ha='center', va='bottom')
    # plt.text(br2[2], task_4_mean/4, "Fill-Level: 70%", ha='center', va='bottom')

    plt.xticks([0+bar_width/2, 1+bar_width/2], ['Container 1', 'Container 2'])
    
    plt.xlabel("Containers")
    plt.ylabel("Trajectory Duration(s)")
    plt.legend()

    plt.show()
    
        