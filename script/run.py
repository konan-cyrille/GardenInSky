#!pip install split_folders
#!pip install tensorflow==1.14.0
#!pip install tensorflow-gpu==1.14
#!pip install imageai --upgrade

import pandas as pd 
import os
import split_folders
import numpy as np
from imageai.Prediction.Custom import ModelTraining





def Run():

    split_folders.ratio('../raw/color', output='../dataSplited')


    #split_folders.ratio('../raw/data', output='dataSplited')

    #split_folders.fixed('../raw/data', output='datS', fixed=(10,10))


    mdl_trainer = ModelTraining()

    mdl_trainer.setModelTypeAsResNet()
    mdl_trainer.setDataDirectory('../dataSplited')
    mdl_trainer.trainModel(num_objects=38, num_experiments=10, batch_size=32, enhance_data=True, show_network_summary=True)



if __name__ == "__main__":
    Run()






