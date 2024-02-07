from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def fit(trainFeatures, trainTargets):
    """

    :param trainFeatures:
    :param trainTargets:
    :return:
    """
    svm = SVC()
    svm.fit(trainFeatures, trainTargets)
    return svm

def testPredict(model, testFeatures):
    prediction = model.predict(testFeatures)
    return prediction

def supVec(trainFeatures, trainTargets, testFeatures, testTargets):
    trainedModel = fit(trainFeatures, trainTargets)
    prediction = testPredict(trainedModel, testFeatures)
    svmAcc = accuracy_score(testTargets, prediction)
    print(f"Support Vector Accuracy: {svmAcc}")
    return svmAcc

