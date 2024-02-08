from dataProcessing import dataPreprocessing as dp
import randomForest as rf
import supportVector as svm
import xgBoost as xgb


# Temp fix to dtype issues
# import warnings
# warnings.filterwarnings("ignore")

# Current ideas for models: Random Forest Classifier, XGBoost, Support Vector Machine, AdaBoost(?)

trainingDataFeatures, trainingDataTargets, testDataFeatures, testDataTargets = dp('dataset.csv')
#
# rf.randForest(trainingDataFeatures, trainingDataTargets, testDataFeatures, testDataTargets)
#
# svm.supVec(trainingDataFeatures, trainingDataTargets, testDataFeatures, testDataTargets)
#
# xgb.xgb(trainingDataFeatures, trainingDataTargets, testDataFeatures, testDataTargets)