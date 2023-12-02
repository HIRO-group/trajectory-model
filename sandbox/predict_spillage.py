import numpy as np
from trajectory_model.classifier_predict_func_api import spilled
from trajectory_model.helper.read import read_panda_vectors
from trajectory_model.spill_free.constants import MAX_TRAJ_STEPS, EMBED_DIM, MOCAP_DT, \
    BIG_DIAMETER, BIG_HEIGHT, SMALL_DIAMETER, SMALL_HEIGHT, \
    BIG_FILL_FULL, BIG_FILL_HALF, SMALL_FILL_FULL, SMALL_FILL_HALF, BLANK_VAL

filenames_props_no_spill = [
    (['01-09-2023 13-42-14', '01-09-2023 13-58-43', '01-09-2023 14-09-56'], 
                        np.array([BIG_DIAMETER, BIG_HEIGHT, BIG_FILL_HALF])),
    (['10-09-2023 10-03-18', '10-09-2023 10-06-37', '10-09-2023 13-10-26', '10-09-2023 13-14-07'],
                        np.array([SMALL_DIAMETER, SMALL_HEIGHT, SMALL_FILL_FULL])),
    (['10-09-2023 13-30-09', '10-09-2023 13-32-29', '10-09-2023 13-39-37'], 
                        np.array([SMALL_DIAMETER, SMALL_HEIGHT, SMALL_FILL_HALF]))
]

if __name__ == "__main__":
    for fps in filenames_props_no_spill:
        filenames = fps[0]
        properties = fps[1]
        for file_name in filenames:
            panda_file_path =  '/home/ava/projects/assets/cartesian/'+file_name+'/cartesian_positions.bin'
            trajectory = read_panda_vectors(panda_file_path)
            spillage_probability = spilled(trajectory, properties)
        print("------")

