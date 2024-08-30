import numpy as np
from trajectory_model.helper.read import read_panda_vectors
from sandbox.constants import Tasks

def get_distance_travelled(filename):
    panda_file_path =  '/home/ava/projects/assets/cartesian/'+filename+'/cartesian_positions.bin'
    panda_trajectory = read_panda_vectors(panda_file_path)
    distance_travelled = 0
    for idx, pos in enumerate(panda_trajectory):
        if idx == 0:
            continue
        distance_travelled += np.linalg.norm(np.array(pos[:3]) - np.array(panda_trajectory[idx-1][:3]))
    
    time = len(panda_trajectory)/1000
    return distance_travelled, time

if __name__ == "__main__":
    times, distance_travelled, vel_aprxs = [], [], []

    for task in [Tasks.task_1, Tasks.task_2, Tasks.task_3, Tasks.task_4]:
        vel = []
        for file_name in task:
            distance_travelled, time = get_distance_travelled(file_name)
            vel.append(distance_travelled/time)
        
        vel_aprxs.append(np.mean(vel))
    
    print(vel_aprxs)
