from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import numpy as np

# Industry standard is 10-fold cross-validation
folds = 10

param_grid = {
    'booster': ['gbtree', 'gblinear', 'dart']
}

featureDropList = ['min_flowiat', 'total_biat']

def gridSearch(model, trainFeature, trainTarget):
    """
    This function Grid Searches a model with cross-validation to determine optimal hyper-parameters
    Args:
        model (SupportVector): Our initial untrained model

    Returns:
        finalModel (SupportVector): Our untrained model with the best calculated hyper-parameters
    """
    gSearch = GridSearchCV(model, param_grid, cv=folds, n_jobs=-1)
    gSearch.fit(trainFeature, trainTarget)

    best_params = gSearch.best_params_
    best_score = gSearch.best_score_

    print(f"Best parameters: {best_params}")
    print(f"Best score: {best_score}")

    tunedModel = gSearch.best_estimator_
    return tunedModel


def fit(trainFeatures, trainTargets):
    """
    This function fits our XGBClassifier

    Args:
        trainFeatures: The features of our training dataset
        trainTargets: The targets of our training dataset
    Returns:
        XGBClassifier: Our fitted XGBClassifier

    """
    xgb = XGBClassifier()
    xgbTuned = gridSearch(xgb, trainFeatures, trainTargets)
    xgbTuned.fit(trainFeatures, trainTargets)
    return xgbTuned

def testPredict(model, testFeatures):
    """
    This function tests our trained model on a given set of test features and outputs the predictions

    Args:
        model: The XGBClassifier model
        testFeatures: The test feature portion of our DataSet

    Returns:
        numpy array: The predicted values of our testFeatures
    """
    prediction = model.predict(testFeatures)
    return prediction

def xgb(trainFeatures, trainTargets, testFeatures, testTargets, encoder):
    print('XGBClassifier:')
    xgbTrainFeatures = trainFeatures.drop(columns=featureDropList)
    xgbTestFeatures = testFeatures.drop(columns=featureDropList)
    trainedModel = fit(xgbTrainFeatures, trainTargets)
    prediction = testPredict(trainedModel, xgbTestFeatures)
    uniqueTargets = encoder.inverse_transform(np.unique(testTargets))
    print(classification_report(testTargets, prediction, target_names=uniqueTargets))
    return 0
