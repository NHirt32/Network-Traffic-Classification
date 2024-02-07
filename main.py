from dataProcessing import dataPreprocessing as dp
import randomForest as rf
import numpy as np
import dataVisualization as dv
import pandas as pd

# Temp fix to dtype issues
import warnings
warnings.filterwarnings("ignore")

# Current ideas for models: Random Forest Classifier, XGBoost, Support Vector Machine, AdaBoost(?)

trainingDataFeatures, trainingDataTargets, testDataFeatures, testDataTargets = dp('dataset.csv')

print(trainingDataTargets.isnull().sum())
print(trainingDataTargets.isin([np.nan, np.inf, -np.inf]).sum())

rf.randForest(trainingDataFeatures, trainingDataTargets, testDataFeatures, testDataTargets)
# trainingData = trainingDataFeatures.insert(len(trainingDataFeatures), 'level', trainingDataTargets.values.ravel)