DATA_DIR = 'data/data_may_30_final.csv'
TRAJECTORY_DURATION = 5 # only used in collection: TODO clean this later

''' 
embedding = (x, o, cup_type) 
x=(x, y, z)
o=(a, b, c, d)
cup_type = 1 or 2 or 3
'''
T_max = 3
dt = 0.02
MAX_TRAJ_STEPS = int(T_max//dt)
NUM_HEADS = 8
FF_DIM = 32
EMBED_DIM = 8

E_ID = 35 # This is used in test




# I'm gonna start with safe trajectories.

# exclude 4, 6, 11, 12, 13, 14, 15, 16, 22, 23, 25, 31

# 21 spilled on wards untill 35


# data_may_30.csv
# 20, upwards (0.6555968973339276, -0.035601866156174916, -179.9728119325014)
# 21, downwards (176.9016086092922, 0.12855806074979775, 0.6519752773257345)
# 22, tilted to the left (34.57178695023557, 45.31732259297031, -163.65679976769698)
# 23, tilted tp the right (-17.874004431561872, -33.62999404428047, -177.96834483185378)



# 28 euler for straight up (110.08930456642592, -4.803311019969231, -128.07267596812795)
# 29 euler for right (-169.72191696191032, 63.93468463465511, -33.80729119912107)
# 30 euler for left (157.3564147145408, -82.92020928394471, 161.86684534223383)
# 31 euler for straight up (110.02948243898217, -5.281399514151082, -130.07743939300624)
# 32 euler for down (-101.96544838950751, -7.93701047060716, 38.4881498294685)

# 33 (-0.0009011504954008363, 0.011689754484071923, 179.99902831367237)
# 34  (-159.28021858420516, -9.14632329258329, -0.71280915065024)