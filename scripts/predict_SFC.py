import numpy as np
from trajectory_model.SFC import get_SFC_model
from trajectory_model.process_data.data_processor import read_panda_trajectory
from trajectory_model.predict_api import process_panda_to_model_input
from trajectory_model.common import get_arguments
from trajectory_model.process_data.containers import WineGlass, FluteGlass, BasicGlass, RibbedCup, TallCup, CurvyWineGlass

def is_spill_free(args, x_traj, x_prop, msg=''):
    model = get_SFC_model()
    model.load_weights(args.load_weight_addr)
    prediction = model.predict({"trajectory": x_traj[None, :, :], "properties": x_prop[None, :],})[0][0]
    print("Spill-Free? ", prediction, ', ', msg)
    return prediction < 0.5


def get_panda_trajectory(file_address):
    trajectory = read_panda_trajectory(file_address)
    trajectory = np.array([np.array(trajectory[i]) for i in range(0, len(trajectory))])
    trajectory = process_panda_to_model_input(trajectory)
    return trajectory


if __name__ == "__main__":
    args = get_arguments()
    args.load_weight_addr = "artifacts/weights/sfc/2024-10-15 11:06:07_acc0.93_loss0.19_epoch395.h5" # 2
                    
    trajectory_file_address = 'data/experiments/task_1/21-11-2023 12-07-50/cartesian.csv'
    trajectory = get_panda_trajectory(trajectory_file_address)
    container = WineGlass(WineGlass.low_fill)
    properties = np.array([container.diameter_b, container.height, container.diameter_u, container.fill_level])
    is_spill_free(args, trajectory, properties)
