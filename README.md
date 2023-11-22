# Maceral_Identification
# Project Description
This project mainly focuses on training different deep learning models, including PSP, deeplabV3+, Unet, ATTUNet, UNetFormer, UNetFormerEDGE, etc. These models are used for semantic segmentation tasks of bimodal image data. Due to confidentiality, the specific dataset cannot be provided.
The project also includes the process of model training and the saving of training results.

This project is still being supplemented and improved (as of 2023.11.10).

# How to Run
First, you need to install all necessary dependencies, which can be done using the following command: pip install -r requirements.txt
Then, you can start training and testing the models by running the main.py file: python main.py

# Results
The results of training and testing will be saved in the "log/" directory, including the best model files (.pkl) and training logs (.json) for each fold of each model.

You can also view the visualized results of training loss, validation loss, and mIOU by running the code.

# File Structure
Here is a description of the functions of various files in the project:

--cut_image1/, cut_image2/, cut_label/: Contain the bimodal dataset and its labels used in this project.

--cut_label_v/: Visualized results of the labels.

--log/: This directory will save the results of training and testing, including the best model files (.pkl) and training logs (.json) for each model.

--_utils.py: Contains some utility functions, such as model selection, training, etc.

--dataset.py: Contains the code related to dataset loading, mainly the Dataset class.

--loss.py: This file contains loss functions used for training models, such as Dice_Loss, etc.

--main.py: This is the main entry of the project, containing the entire training process of the model and parameter settings, etc.

--pre_processing.ipynb: This file contains code related to data preprocessing, such as image segmentation, enhancement, etc.

--README.md: This file contains the introduction and usage instructions of the project.

--requirements.txt: This file lists all the dependencies required to run the project.

--score.py: This file contains scoring functions to evaluate the performance of the model.

--UNetFormer.py: This file contains the implementation code of the UnetFormer and UnetFormerEdge models.

#  Contact us: 
liangzou@cumt.edu.cn  ronghuanzhao@cumt.edu.cn
