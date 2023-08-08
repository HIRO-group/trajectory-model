'''
embedding = (sampled_so_far, cup_type)
sampled_so_far [(x, y, z, a, b, c, w), ... ]
cup_type = 1 or 2 or 3 ...
'''

MAX_NUM_WAYPOINTS = 10
EMBED_DIM_POS = 4  # pos, cup_type for now
NUM_HEADS = 8
FF_DIM = 32
