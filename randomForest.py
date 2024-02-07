from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def fit(trainFeatures, trainTargets):
    """

    :param trainFeatures:
    :param trainTargets:
    :return:
    """
    rfc = RandomForestClassifier()
    rfc.fit(trainFeatures, trainTargets.values.ravel())
    return rfc

def testPredict(model, testFeatures):
    prediction = model.predict(testFeatures)
    return prediction

def randForest(trainFeatures, trainTargets, testFeatures, testTargets):
    trainedModel = fit(trainFeatures, trainTargets)
    prediction = testPredict(trainedModel, testFeatures)
    rfcAccuracy = accuracy_score(testTargets, prediction)
    print(rfcAccuracy)
    return rfcAccuracy

