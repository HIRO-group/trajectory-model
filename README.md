## Train SFC
```python main_SFC.py```

## Predict from SFC
 call func spilled(trajectory) from ```trajectory_model/classifier_predict.py``` OR
 ```python helper/predict_spillage.py```

## Process data collected for SFC
```python process_data/process_data_SFC.py```

## Visualize data collected from mocap
``` python helper/draw_file.py```


## Visualize cartesian positions from panda 
``` python helper/draw_panda_path.py```


## Visualize cup orientation and rotated panda ee orientaion
This file also contains the equations for diff * panda_ee = cup

``` python helper/draw_panda_vs_cup.py```


