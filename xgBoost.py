from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def fit(trainFeatures, trainTargets):
    """

    :param trainFeatures:
    :param trainTargets:
    :return:
    """
    xgb = XGBClassifier()
    xgb.fit(trainFeatures, trainTargets)
    return xgb

def testPredict(model, testFeatures):
    prediction = model.predict(testFeatures)
    return prediction

def xgb(trainFeatures, trainTargets, testFeatures, testTargets):
    trainedModel = fit(trainFeatures, trainTargets)
    prediction = testPredict(trainedModel, testFeatures)
    xgbAccuracy = accuracy_score(testTargets, prediction)
    print(f"XGB Classifier accuracy: {xgbAccuracy}")
    return xgbAccuracy
