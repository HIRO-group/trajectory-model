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

'''All the numbers are in inches'''
# original wine glass:
BIG_DIAMETER_B = 3
BIG_HEIGHT = 4
BIG_DIAMETER_U = 3
BIG_FILL_80 = 0.8
BIG_FILL_30 = 0.3

# original flute glass:
SMALL_DIAMETER_B = 0.5
SMALL_HEIGHT = 5
SMALL_DIAMETER_U = 1.8
SMALL_FILL_80 = 0.8
SMALL_FILL_50 = 0.5


# short tumbler:
SHORT_TUMBLER_DIAMETER_B = 2.7
SHORT_TUMBLER_HEIGHT = 3.4
SHORT_TUMBLER_DIAMETER_U = 3.2
SHORT_TUMBLER_FILL_30 = 0.3
SHORT_TUMBLER_FILL_70 = 0.7

# tall tumbler:
TALL_TUMBLER_DIAMETER_B = 2.4
TALL_TUMBLER_HEIGHT = 6
TALL_TUMBLER_DIAMETER_U = 3
TALL_TUMBLER_FILL_50 = 0.5
TALL_TUMBLER_FILL_80 = 0.8

# tumbler:
TUMBLER_DIAMETER_B = 2.5
TUMBLER_HEIGHT = 3.7
TUMBLER_DIAMETER_U = 3.2
TUMBLER_FILL_30 = 0.3
TUMBLER_FILL_70 = 0.7

# wine:
WINE_DIAMETER_B = 2
WINE_HEIGHT = 3.8
WINE_DIAMETER_U = 2.9
WINE_FILL_30 = 0.3
WINE_FILL_70 = 0.7


class FLASK:
    diameter_u = 1
    diameter_b = 3
    height = 5
    fill_level_60 = 0.6
    fill_level_90 = 0.9

class BEAKER:
    diameter_u = 2
    diameter_b = 2
    height = 3
    fill_level_50 = 0.5
    fill_level_90 = 0.9

class CYLINDER:
    diameter_u = 1
    diameter_b = 1
    height = 7
    fill_level_20 = 0.2
    fill_level_80 = 0.8
