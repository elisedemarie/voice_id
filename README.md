# Voice ID
*a repo for identifying certain characteristics about voices from mp3 inputs.*

## Purpose
This project makes use of the Mozilla Common Voice public dataset. The project aims to be a lightweight approach to processing voice data, building relevant features, and making predictions about the speaker's self-reported gender and age. 


## How it Works
This project makes use of signal processing to engineer features from the raw signal data. The model processes several features of sound analysis relevant to vocal characteristics and creates a feature set for each sound across the signal. This signal is then broken up into small windows. Each window's individual features are pooled across the window. The final result outputs the min, max, and mean of each feature for each window. 

The individual segments are then used to train a classifier that learns to make predictions. Once the model has been trained, predictions are made for every segment. For each original sound file, the majority prediction across the window is output as the prediction for that sound file.

![voice_id_diagram drawio (1)](https://github.com/elisedemarie/voice_id/assets/135685125/fcf91ef9-9954-4e20-a9a7-0303ed820da1)

## How to Use
The module for this project is broken up into several scripts but is easy to use. The main modifications users need make is to the config.json file. This file contains the links to the relevant directories and files for the project.

The module is broken up into three stages that can be run together or separately. Preparing the data, training the model, and evaluating. Below is a guide for each.

### 1. Preparing the Data
To prepare the data, you should have first downloaded a dataset from the Common Voice website. This dataset needs to be unzipped and then two links put in the config.json file. The location of the "validation.tsv" file contains the list of all validated files. Paste the path of this file into the config.json file under "validation_tsv". The second path is the location to the clips. This is the directory containing all the .mp3 files listed in the validation.tsv file. Paste this path under "data_dir".

Before running, also update the "output_dir" to your desired output for the pipeline. The script will output multiple .pkl files that ease re-running the pipeline and running in segments.

Once all locations have been updated in the config.json file you can run the first step. To only run this step run the command `python3 main.py prepare`

## 2. Training the Model
Once the data has been prepared you can train the model. Ensure that the directories in the config.json are correct. There are two models available for training. The first is an XGBoost Classifier, the second a Multi-Layer Perceptron. To run the XGB model use the command `python3 main.py train -xgb`. To run the MLP use the command `python3 main.py train -mlp`. 

The output of training will save the models and print the result. **NOTE** This result is only the performance of the model on individual segments. The pooled results are shown in evaluation.

## 3. Evaluating
To evaluate the model, ensure all directories in config.json are correct then select which set to evaluate on. If you want to evaluate on the validation set use the command `python3 main.py eval -val`. If you want to evaluate on the testing set use the command `python3 main.py eval -test`. 

Evaluation results are output as a classification report. The 0 class represents male self-reported voices and the 1 class female. These results are the majority prediction of the segments for each mp3 file.


