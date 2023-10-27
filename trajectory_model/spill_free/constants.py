''' 
embedding = (x, o, radius, height, fill_level)
x=(x, y, z)
o=(a, b, c, w)
'''

NUM_HEADS = 8
FF_DIM = 32
MAX_TRAJ_STEPS = 150
EMBED_DIM = 10

MOCAP_DT = 6
ROBOT_DT = 50

''' 
mocap freq is 120HZ, robot is 1000HZ
we want them to have the same frequency
MOCAP_DT = 6 and ROBOT_DT = 50 because:
120/6 = 20 and 1000/50 = 20
'''


BLANK_VAL = 1000


BIG_RADIUS = 3
BIG_HEIGHT = 4
SMALL_RADIUS = 1.8
SMALL_HEIGHT = 5

BIG_FILL_FULL = 0.8
BIG_FILL_HALF = 0.3

SMALL_FILL_FULL = 0.8
SMALL_FILL_HALF = 0.5