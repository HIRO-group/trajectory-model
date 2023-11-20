''' 
embedding = (x, o, radius_b, height, radius_u, fill_level)
x=(x, y, z)
o=(a, b, c, w)
'''
MAX_TRAJ_STEPS = 150
EMBED_DIM = 11

EMBED_LOC = 7 # x, y, z, a, b, c, w
EMBED_PROP = 4 # radius_buttom, height, radius_up, fill_level

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
BIG_RADIUS_B = 3
BIG_HEIGHT = 4
BIG_RADIUS_U = 3
BIG_FILL_80 = 0.8
BIG_FILL_30 = 0.3

# original flute glass:
SMALL_RADIUS_B = 0.5
SMALL_HEIGHT = 5
SMALL_RADIUS_U = 1.8
SMALL_FILL_80 = 0.8
SMALL_FILL_50 = 0.5


# short tumbler:
SHORT_TUMBLER_RADIUS_B = 2.7
SHORT_TUMBLER_HEIGHT = 3.4
SHORT_TUMBLER_RADIUS_U = 3.2
SHORT_TUMBLER_FILL_30 = 0.3
SHORT_TUMBLER_FILL_70 = 0.7

# tall tumbler:
TALL_TUMBLER_RADIUS_B = 2.4
TALL_TUMBLER_HEIGHT = 6
TALL_TUMBLER_RADIUS_U = 3
TALL_TUMBLER_FILL_50 = 0.5
TALL_TUMBLER_FILL_80 = 0.8

# tumbler:
TUMBLER_RADIUS_B = 2.5
TUMBLER_HEIGHT = 3.7
TUMBLER_RADIUS_U = 3.2
TUMBLER_FILL_30 = 0.3
TUMBLER_FILL_70 = 0.7

# wine:
WINE_RADIUS_B = 2
WINE_HEIGHT = 3.8
WINE_RADIUS_U = 2.9
WINE_FILL_30 = 0.3
WINE_FILL_70 = 0.7