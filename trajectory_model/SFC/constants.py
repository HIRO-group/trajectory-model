''' 
embedding = (x, o, radius_b, height, radius_u, fill_level)
x=(x, y, z)
o=(a, b, c, w)
'''
MAX_TRAJ_STEPS = 150
EMBED_DIM = 11

EMBED_LOC = 7 # x, y, z, a, b, c, w
EMBED_PROP = 4 # diameter_buttom, height, diameter_up, fill_level

MOCAP_DT = 6
ROBOT_DT = 50
BLANK_VAL = 1000

''' 
mocap freq is 120HZ, robot is 1000HZ
we want them to have the same frequency
MOCAP_DT = 6 and ROBOT_DT = 50 because:
120/6 = 20 and 1000/50 = 20
'''