from read import read_vectors
from draw_file import read_file

file_name = '01-09-2023 13-42-14'
# file_name = '01-09-2023 13-58-43'
# file_name = "01-09-2023 14-09-56"

panda_file_path =  '/home/ava/projects/assets/cartesian/'+file_name+'/cartesian_positions.bin'
panda_trajectory = read_vectors(panda_file_path)

cup_file_path ="/home/ava/projects/trajectory-model/data/mocap_new/big/full/spill-free/2023-09-08 20:24:51.csv"
cup_trajectory = read_file(cup_file_path)