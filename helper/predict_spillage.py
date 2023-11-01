from trajectory_model.classifier_predict_func_api import spilled
from read import read_vectors

file_name = '01-09-2023 13-42-14'
# file_name = '01-09-2023 13-58-43'
# file_name = "01-09-2023 14-09-56"

panda_file_path =  '/home/ava/projects/assets/cartesian/'+file_name+'/cartesian_positions.bin'
trajectory = read_vectors(panda_file_path)

spillage_probability = spilled(trajectory)

