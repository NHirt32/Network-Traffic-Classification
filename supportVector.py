from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import numpy as np

# Industry standard is 10-fold cross-validation
folds = 10

param_grid = {
    'kernel': ['rbf', 'poly'],
    'degree': [1,2,3,4,5,6,7,8,9,10],
    'gamma': ['scale', 'auto']

}

featureDropList = ['min_flowiat']

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
    This function fits our Support Vector Classifier

    Args:
        trainFeatures: The features of our training dataset
        trainTargets: The targets of our training dataset
    Returns:
        SupportVectorClassifier: Our fitted Support Vector Classifier

    """
    svm = SVC()
    tunedSVM = gridSearch(svm, trainFeatures, trainTargets)
    tunedSVM.fit(trainFeatures, trainTargets)
    return tunedSVM

def testPredict(model, testFeatures):
    """
    This function tests our trained model on a given set of test features and outputs the predictions

    Args:
        model: The SVC model
        testFeatures: The test feature portion of our DataSet

    Returns:
        numpy array: The predicted values of our testFeatures
    """
    prediction = model.predict(testFeatures)
    return prediction

def supVec(trainFeatures, trainTargets, testFeatures, testTargets, encoder):
    print('Support Vector Classifier:')
    svmTrainFeatures = trainFeatures.drop(columns=featureDropList)
    svmTestFeatures = testFeatures.drop(columns=featureDropList)
    trainedModel = fit(svmTrainFeatures, trainTargets)
    prediction = testPredict(trainedModel, svmTestFeatures)
    uniqueTargets = encoder.inverse_transform(np.unique(testTargets))
    print(classification_report(testTargets, prediction, target_names=uniqueTargets))
    return 0

