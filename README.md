
# 2024 RPL Fall Project:  Optimal Dynamic 3D Reconstruction using Moving Monocular Camera Mounted on Robotic Arm

## System Environments
- Ubuntu 22.04
- ROS2 Humble

## Install ROS2
Follow the prompts to install ROS2 Humble

```
cd /tmp && wget http://fishros.com/install -O fishros && . fishros
```

## For Simulator
### Install
Please ref this webside to install gazebo and UR5：
https://blog.csdn.net/Vittore_Li/article/details/138697480

### Build
```
cd ur5e_ws
colcon build
source install/setup.bash
ros2 launch ur_simulation_gazebo ur_sim_moveit.launch.py
```
Now you can see the 2 windows(rviz2 and gazebo)

## For Neu-NBV
### Install
```
cd scripts
pip install -r requirements.txt
```
- Download [DTU dataset](https://phenoroam.phenorob.de/file-uploader/download/public/953455041-dtu_dataset.zip) dataset to `scripts/neural_rendering/data/dataset` folder.
- Here is the pretrained models trained on [DTU](https://phenoroam.phenorob.de/file-uploader/download/public/195880506-dtu_training.zip) Copy these folder to `scripts/neural_rendering/logs`.


### Run
```
# In Terminal1
python3 scripts/planning/ik_solver.py
# In Terminal2
python3 scripts/planning/tm5_experiment.py
```
The image will be saved in `./scripts/experiments/TM5_test`
