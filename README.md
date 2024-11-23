## AE/ME 7785 IRR Lab 6 Part 1 - Vision Checkpoint
### (The repository for the second part of this lab, i.e. the final robot demo, is [lab6_final](https://github.com/gtrobo/lab6_final)) 
*(Assumes a base install of Ubuntu 22 and Python3)*
1. Navigate to your desired working directory (not the ROS workspace). You may default to the home directory: `cd $HOME` .
2. Create a Python virtual environment and activate it: `python3 -m venv vision33` and `source vision33/bin/activate` .
3. Clone this repository inside the virtual environment subdirectory: `cd vision33` and then `git clone git@github.com:gtrobo/7785-lab6-ml.git`.
4. Install Python libraries and dependencies: `pip install -r 7785-lab6-ml/requirements.txt`. Pip may take a few minutes to complete installing all the modules.
5. Inspect the contents of the 7785-lab6-ml directory to ensure that the following two files exist among others: `cd 7785-lab6-ml` and look for `vision33_classifier.pkl` and `model_tester.py`.
6. Open the model_tester code in a text editor or IDE of your choice: `gedit model_tester.py` or `code model_tester.py`.
7. Edit line 13 of this code and supply the path of the test dataset folder. Save and exit.
8. Before running the model tester code, we need to ensure that the test data folder is prepared with subfolders named 0, 1, 2, 3, 4, and 5, correspoding to the labels of the images. 
    - The `move_images.sh` script will do this task. Copy it into the test dataset folder: `cp move_images.sh /home/sean/pythonGrading/2024F_Gimgs`
    - Make it an executable: `chmod +x move_images.sh`
    - Run the script: `./move_images.sh`
    - Inspect the dataset folder for the newly created subdirectories named 0 - 5 with PNG files inside each.
9. Go back to the 7785-lab6-ml directory: `cd $HOME/vision33/7785-lab6-ml` and run the tester code: `python3 model_tester.py`. Ignore the FutureWarning on pickle module. Prediction labels, accuracy, and classification report are printed to the terminal. A pop-up window will contain the confusion matrix. 
10. To exit, close the pop-up, Ctrl + C in the terminal, and `deactivate` the virtual environment.
________

![Confusion Matrix](/ConfusionMatrix.png)
________
#### Group 33: Pratheek Manjunath, Chris Meier, GLTron.<br>
pratheek.m@gatech.edu
