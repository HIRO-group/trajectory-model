'''
embedding = (sampled_so_far, cup_type)
sampled_so_far [(x, y, z, a, b, c, w), ... ]
cup_type = 1 or 2 or 3 ...
'''

MAX_NUM_WAYPOINTS = 10

EMBED_DIM_POS_X = 4 # pos, cup_type for now
EMBED_DIM_POS_Y = 3 # pos

EMBED_DIM_ORIENTATION_X = 8 # pos, a, b, c, w, cup_type for now
EMBED_DIM_ORIENTATION_Y = 4 # a, b, c, w
