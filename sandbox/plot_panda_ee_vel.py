from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import numpy as np

from trajectory_model.helper.read import read_panda_vectors
from trajectory_model.helper.jacobian import get_jacobian
from sandbox.constants import Tasks


def get_ee_vel(file_name):
    file_path_v = '/home/ava/projects/assets/'+file_name+'/joint_velocities.bin'
    file_path_p = '/home/ava/projects/assets/'+file_name+'/joint_waypoints.bin'
    joint_velocities = read_panda_vectors(file_path_v)
    joint_poisitions = read_panda_vectors(file_path_p)

    ee_velocities = []
    for idx, vel in enumerate(joint_velocities):
        J = get_jacobian(joint_poisitions[idx])
        ee_vel = J.dot(vel)
        ee_velocities.append(ee_vel)

    x_vels, y_vels, z_vels = [], [], []
    vels = []
    for vel in ee_velocities:
        x_vels.append(vel[0])
        y_vels.append(vel[1])
        z_vels.append(vel[2])
        vels.append(np.linalg.norm(vel))
    
    return x_vels, y_vels, z_vels, vels


if __name__ == "__main__":
    # vels_task_1, vels_task_2 = [], []
    # vels_task_3, vels_task_4 = [], []
    # vels_task_5, vels_task_6 = [], []

    # for file_name in Tasks.task_1:
    #     _, _, _, vels = get_ee_vel(file_name)
    #     max_mean_median = [np.max(vels), np.mean(vels), np.median(vels), np.std(vels)]
    #     vels_task_1.append(max_mean_median)

    # for file_name in Tasks.task_2:
    #     _, _, _, vels = get_ee_vel(file_name)
    #     max_mean_median = [np.max(vels), np.mean(vels), np.median(vels), np.std(vels)]
    #     vels_task_2.append(max_mean_median)

    # for file_name in Tasks.task_3:
    #     _, _, _, vels = get_ee_vel(file_name)
    #     max_mean_median = [np.max(vels), np.mean(vels), np.median(vels), np.std(vels)]
    #     vels_task_3.append(max_mean_median)
    
    # for file_name in Tasks.task_4:
    #     _, _, _, vels = get_ee_vel(file_name)
    #     max_mean_median = [np.max(vels), np.mean(vels), np.median(vels), np.std(vels)]
    #     vels_task_4.append(max_mean_median)
    
    # for file_name in Tasks.task_5:
    #     _, _, _, vels = get_ee_vel(file_name)
    #     max_mean_median = [np.max(vels), np.mean(vels), np.median(vels), np.std(vels)]
    #     vels_task_5.append(max_mean_median)

    # for file_name in Tasks.task_6:
    #     _, _, _, vels = get_ee_vel(file_name)
    #     max_mean_median = [np.max(vels), np.mean(vels), np.median(vels), np.std(vels)]
    #     vels_task_6.append(max_mean_median)


    # task_1_max = np.max([sublist[0] for sublist in vels_task_1])
    # task_1_mean = np.mean([sublist[1] for sublist in vels_task_1])
    # task_1_std = np.mean([sublist[3] for sublist in vels_task_1])

    # task_2_max = np.max([sublist[0] for sublist in vels_task_2])
    # task_2_mean = np.mean([sublist[1] for sublist in vels_task_2])
    # task_2_std = np.mean([sublist[3] for sublist in vels_task_2])

    # task_3_max = np.max([sublist[0] for sublist in vels_task_3])
    # task_3_mean = np.mean([sublist[1] for sublist in vels_task_3])
    # task_3_std = np.mean([sublist[3] for sublist in vels_task_3])

    # task_4_max = np.max([sublist[0] for sublist in vels_task_4])
    # task_4_mean = np.mean([sublist[1] for sublist in vels_task_4])
    # task_4_std = np.mean([sublist[3] for sublist in vels_task_4])

    # task_5_max = np.max([sublist[0] for sublist in vels_task_5])
    # task_5_mean = np.mean([sublist[1] for sublist in vels_task_5])
    # task_5_std = np.mean([sublist[3] for sublist in vels_task_5])

    # task_6_max = np.max([sublist[0] for sublist in vels_task_6])
    # task_6_mean = np.mean([sublist[1] for sublist in vels_task_6])
    # task_6_std = np.mean([sublist[3] for sublist in vels_task_6])

    # print("big 80 1 max: ", task_1_max)   
    # print("big 30 1 max: ", task_2_max)
    # print("small 80 1 max: ", task_3_max)
    # print("small 50 1 max: ", task_4_max)
    # print("tumbler 70 1 max: ", task_5_max)
    # print("tumbler 30 1 max: ", task_6_max)

    # print("big 80 1 mean: ", task_1_mean)
    # print("big 30 1 mean: ", task_2_mean)
    # print("small 80 1 mean: ", task_3_mean)
    # print("small 50 1 mean: ", task_4_mean)
    # print("tumbler 70 1 mean: ", task_5_mean)
    # print("tumbler 30 1 mean: ", task_6_mean)

    # print("big 80 1 std: ", task_1_std)
    # print("big 30 1 std: ", task_2_std)
    # print("small 80 1 std: ", task_3_std)
    # print("small 50 1 std: ", task_4_std)
    # print("tumbler 70 1 std: ", task_5_std)
    # print("tumbler 30 1 std: ", task_6_std)

    # raise Exception("stop")

    # task_1_mean = 0.33865465414511353
    # task_2_mean = 0.4857333738024513
    # task_3_mean = 0.5903238711670706
    # task_4_mean = 0.608259862076469
    # task_5_mean = 0.42316489634116633
    # task_6_mean = 0.5982179386889576

    
    task_1_mean =  0.33865465414511353
    task_2_mean =  0.4857333738024513
    task_3_mean =  0.5903238711670706
    task_4_mean =  0.608259862076469
    task_5_mean =  0.42316489634116633
    task_6_mean =  0.5982179386889576

    task_1_std  =   0.12220735153533106
    task_2_std =  0.13378073068856072
    task_3_std = 0.16140427249361788
    task_4_std = 0.1699177766264402
    task_5_std = 0.13979103552681185
    task_6_std = 0.17810548993580633
    
    fig = plt.figure()
    bar_width = 0.4
    br1 = np.arange(3)
    br2 = [x + bar_width for x in br1]

    plt.bar(br1, [task_1_mean, task_3_mean, task_5_mean], \
            yerr=[task_1_std, task_3_std, task_5_std], \
            color='#7FBFFF', width = bar_width, edgecolor ='#7FBFFF')

    plt.bar(br2, [task_2_mean, task_4_mean, task_6_mean], \
            yerr=[task_2_std, task_4_std, task_6_std], \
            color='#4d79bc', width = bar_width, edgecolor ='#4d79bc')
    
    
    # plt.text(br1[0], task_4_mean/4, f'std: {round(task_1_std, 2)}', ha='center', va='bottom', color='white')
    # plt.text(br1[1], task_4_mean/4, f'std: {round(task_3_std, 2)}', ha='center', va='bottom', color='white')
    # plt.text(br1[2], task_4_mean/4, f'std: {round(task_5_std, 2)}', ha='center', va='bottom', color='white')
    # plt.text(br2[0], task_4_mean/4, f'std: {round(task_2_std, 2)}', ha='center', va='bottom', color='white')
    # plt.text(br2[1], task_4_mean/4, f'std: {round(task_4_std, 2)}', ha='center', va='bottom', color='white')
    # plt.text(br2[2], task_4_mean/4, f'std: {round(task_6_std, 2)}', ha='center', va='bottom', color='white')

    plt.xticks([0+bar_width/2, 1+bar_width/2, 2+bar_width/2], ['Wine Glass', 'Flute Glass', 'Basic Glass'])
    
    # plt.xlabel("Containers")
    plt.ylabel("Mean Velocity")
    plt.legend()

    plt.show()


