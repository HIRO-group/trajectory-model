## Setup
```
git clone git@github.com:HIRO-group/trajectory-model.git
conda activate sfrrt
cd trajectory-model
pip install -e .
```

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
