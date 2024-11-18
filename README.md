## AE/ME 7785 IRR Lab 6 Part 1 - Vision Checkpoint
### (The repository for the second part of this lab, i.e. the final robot demo, is [lab6_final](https://github.com/gtrobo/lab6_final)) 
*(Assumes a base install of Ubuntu 22 and Python3)*
1. Navigate to your desired working directory (not the ROS workspace). The default path is `cd $HOME` 
2. Create a Python virtual environment and activate it: `python3 -m venv vision33` and `source vision33/bin/activate`.
3. Clone this repository inside the virtual environment subdirectory: `cd vision33` and then `git clone git@github.com:gtrobo/7785-lab6-ml.git`.
4. Install Python libraries and dependencies: `pip install -r 7785-lab6-ml/requirements.txt`. Pip may take a few minutes to complete installing all the modules.
5. Inspect the contents on the 7785-lab6-ml directory to ensure that the following two files exist among others: `vision33_classifier.pkl` and `model_tester.py`.
6. Open the model_tester code in a text editor or IDE of your choice: `gedit model_tester.py` or `code model_tester.py`
7. Edit line 80 of this code and supply the path of the test dataset folder. Save and exit.
8. In the same directory, run the tester code: `python3 model_tester.py`. Prediction labels, accuracy, and classification report are printed to the terminal.
9. To exit out of the virtual environment, use the command `deactivate`.
 
#### Group 33: Pratheek Manjunath, Chris Meier, GLTron.<br>
pratheek.m@gatech.edu<br>
