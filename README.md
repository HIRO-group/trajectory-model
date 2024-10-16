# Clutter-Aware Spill-Free Liquid Transport via Learned Dynamics
This is the repository for training the Spill-Free Classifier (SFC) model presented in the *Clutter-Aware Spill-Free Liquid Transport via Learned Dynamics* paper ([link](https://arxiv.org/abs/2408.00215)).

## Data 
[Data Directory](https://o365coloradoedu-my.sharepoint.com/:f:/g/personal/avab4907_colorado_edu/EhLxWaRCAXVNhUXhQXufFnwBKscaD8d4D3sEeYsNtqN_KA?e=ipJLA8)
- [Mocap](https://o365coloradoedu-my.sharepoint.com/:f:/g/personal/avab4907_colorado_edu/EhLxWaRCAXVNhUXhQXufFnwBKscaD8d4D3sEeYsNtqN_KA?e=ipJLA8): Data collected from individuals handling different container scenarios captured using a motion capture system
- [Panda](https://o365coloradoedu-my.sharepoint.com/:f:/g/personal/avab4907_colorado_edu/EhLxWaRCAXVNhUXhQXufFnwBKscaD8d4D3sEeYsNtqN_KA?e=ipJLA8): Data collected from the panda robotic arm handling different container scenarios. *(Some files are currently missing)*

## Setup
```
git clone git@github.com:HIRO-group/trajectory-model.git
conda create -n "sfrrt"
conda activate sfrrt
cd trajectory-model
pip install -e .
mkdir data/
```
Copy the downloaded data in the `data/` dir to execute the following commands. 

## Train Spill-Free Classifer (SFC)
```python scripts/train_SFC.py```

## Predict Using SFC
```python scripts/predict_SFC.py```

## Calculate Container Tilt Angle
```python scripts/calculate_tilt_angle.py```

## Visualize Mocap Trajectories
```python scripts/plot_mocap_trajectory.py```

## Visualize Panda Trajectories
```python scripts/plot_panda_trajectory.py```

## Consume Data Streamed By Natnet from Mocap
```python scripts/consume_data_from_mocap.py```

## Convert Joint Angles to Cartesian
```python scripts/convert_panda_joints_to_cartesian.py```

## Citation
```
@inproceedings{abderezaei2024clutterawarespillfreeliquidtransport,
  title={Clutter-Aware Spill-Free Liquid Transport via Learned Dynamics}, 
  author={Abderezaei, Ava and Pasricha, Anuj and Klausenstock, Alex and Roncone, Alessandro},
  year={2024},
  booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
}
```
