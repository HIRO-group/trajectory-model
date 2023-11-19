## Train SFC
```python main_SFC_func_api.py```

## Predict from SFC
call func spilled(trajectory) from ```trajectory_model/classifier_predict.py``` 
 
or

 ```python sandbox/predict_spillage.py```

## Process data collected for SFC
```python process_data/process_data_SFC.py```

## Visualize data collected from mocap
``` python sandbox/draw_mocap_file.py```

## Visualize cartesian positions from panda 
``` python sandbox/draw_panda_path.py```

## Visualize cup orientation and rotated panda ee orientaion
``` python sandbox/draw_panda_vs_cup.py```

## Visualize cup trajectory vs rotated panda trajectory
``` python sandbox/draw_panda_vs_cup.py```

## Visualize probability distribution map
``` python sandbox/draw_probability_distribution_map.py```

## How to rotate panda ee frame to be in cup fram
The equations are in this file: ```trajectory_model/helper/rotate_quaternion.py```
