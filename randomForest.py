from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import numpy as np

# Industry standard is 10-fold cross-validation
folds = 10

param_grid = {
    'n_estimators': [50,75,100,150],
    'criterion': ['gini', 'entropy', 'log_loss'],
}

featureDropList = ['min_flowiat', 'total_biat', 'total_fiat', 'min_biat', 'max_biat', 'flowPktsPerSecond', 'min_active', 'max_active']

def gridSearch(model, trainFeature, trainTarget):
    """
    This function Grid Searches a model with cross-validation to determine optimal hyper-parameters
    Args:
        model (RandomForestClassifier): Our initial untrained model

    Returns:
        finalModel (RandomForestClassifier): Our untrained model with the best calculated hyper-parameters
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
    This function fits our Random Forest classifier

    Args:
        trainFeatures: The features of our training dataset
        trainTargets: The targets of our training dataset
    Returns:
        RandomForestClassifier: Our fitted Random Forest Classifier

    """
    rfc = RandomForestClassifier()
    tunedRFC = gridSearch(rfc, trainFeatures, trainTargets)
    tunedRFC.fit(trainFeatures, trainTargets)
    return tunedRFC



def testPredict(model, testFeatures):
    """
    This function tests our trained model on a given set of test features and outputs the predictions

    Args:
        model: The Random Forest Classifier model
        testFeatures: The test feature portion of our DataSet

    Returns:
        numpy array: The predicted values of our testFeatures
    """
    prediction = model.predict(testFeatures)
    return prediction

def randForest(trainFeatures, trainTargets, testFeatures, testTargets, encoder):
    print('Random Forest Classifier:')
    rfcTrainFeatures = trainFeatures.drop(columns=featureDropList)
    rfcTestFeatures = testFeatures.drop(columns=featureDropList)
    trainedModel = fit(rfcTrainFeatures, trainTargets)
    prediction = testPredict(trainedModel, rfcTestFeatures)
    uniqueTargets = encoder.inverse_transform(np.unique(testTargets))
    print(classification_report(testTargets, prediction, target_names=uniqueTargets))
    return trainedModel

