import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from trajectory_model.sampler_predict_new import probability_distribution_map

if __name__ == "__main__":
    print(probability_distribution_map.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.scatter(probability_distribution_map[:, 0], probability_distribution_map[:, 1], probability_distribution_map[:, 2], s=0.1)
    plt.show()