## AE/ME 7785 IRR Lab 6. 
*(Assumes a base install of Ubuntu 22 and Python3)*
1. Navigated to your ROS 2 (colcon) workspace and clone this repository. The default path is `cd $HOME/ros2_ws/src` and `git clone git@github.com:gtrobo/lab6.git`.
2. Install Python libraries and dependencies: `cd $HOME/ros2_ws/src/lab6/resource` and `pip install -r requirements.txt`. Pip may take a few minutes to complete installing all the modules.
3. 
4. Build the colcon workspace: `cd ~/ros2_ws && colcon build --symlink-install` and `source ~/.bashrc`
5. Launch the nodes: `ros2 launch lab6 lab6_launch.py`

#### Group 33: Pratheek Manjunath, Chris Meier, GLTron.<br>
pratheek.m@gatech.edu<br>