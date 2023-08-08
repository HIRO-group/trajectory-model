''' 
embedding = (x, o, cup_type) 
x=(x, y, z)
o=(a, b, c, w)

cup_type = 1 or 2 or 3 -> this should not be coupled with the same embedding
should change the model architecture 
'''
T_max = 3
DT = 0.02
MAX_TRAJ_STEPS = int(T_max/DT)  # 150
NUM_HEADS = 8
FF_DIM = 32
EMBED_DIM = 8

data_name = 'data_may_30_final'
FINAL_MOCAP_RAW_DATA_DIR = f'data/raw/{data_name}.csv'
FINAL_OMPL_RAW_DATA_DIR = f'data/panda_ompl'

MAX_NUM_WAYPOINTS = 10 # Including first and last point