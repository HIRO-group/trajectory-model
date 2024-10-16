from trajectory_model.process_data.data_processor import read_a_file
from trajectory_model.SFC.constants import BLANK_VAL
from plot_panda_trajectory import plot_quivers, get_start_end_points

if __name__ == "__main__":
    file_address = 'data/mocap/basic_glass/30/spill-free/2023-11-16 12:10:37.csv'
    trajectory_w_fake_properties = read_a_file(file_address, 0, 0, 0, 0)

    trajectory = trajectory_w_fake_properties[0, : , 0:7]
    trajectory = [tr for tr in trajectory if tr[0] < BLANK_VAL]

    start_points, end_points = get_start_end_points(trajectory)
    plot_quivers(start_points=start_points, end_points=end_points, quiver_length=0.02)