import pandas as pd
from sklearn.preprocessing import LabelEncoder
import dataVisualization as dv

pd.set_option('display.max_columns', None)

nonNumericColumns = ['level']

def fileRead(trainFile):
    """
    This function converts two .csv files into pandas dataframes

    Args:
        trainFile (.csv): Relative file path of training .csv file
        testFile (.csv): Relative file path of testing .csv file
    Returns:
        tuple: Returns a tuple containing both .csv files transformed into pandas dataframes
    """
    trainingData = pd.read_csv(trainFile)
    return trainingData

def encoding(trainingData, testData):
    """
    This function encodes non-numeric columns of two dataframes into numeric columns

    Args:
        trainingData (DataFrame): Training data of machine learning model in Pandas DataFrame format
        testData (DataFrame): Test Data for machine learning model in Pandas DataFrame format
    Returns:
        tuple: Returns a tuple of training and testing dataframes with non-numeric data encoded into numeric data
    """
    le = LabelEncoder()
    for column in nonNumericColumns:
        trainingData.loc[:, column] = le.fit_transform(trainingData[column])
        testData.loc[:, column] = le.fit_transform(testData[column])
    return trainingData, testData

def scaling(trainingData, testData, scalableFeatureList):
    """
    This function utilizes min-max scaling to scale all numeric values in our two dataframes

    Args:
        trainingData (DataFrame): Training data of machine learning model in Pandas DataFrame format
        testData (DataFrame): Test Data for machine learning model in Pandas DataFrame format
    Returns:
        tuple: Returns a tuple of training and testing dataframes where all numeric columns have been scaled between 0 - 1
    """
    for feature in scalableFeatureList:
        trainingMax = trainingData[feature].max()
        testMax = testData[feature].max()
        trainingMin = trainingData[feature].min()
        testMin = testData[feature].min()
        trainingData.loc[:, feature] = (trainingData[feature] - trainingMin) / (trainingMax - trainingMin)
        testData.loc[:, feature] = (testData[feature] - testMin) / (testMax - testMin)
    return trainingData, testData

def trainingSplit(splitRatio, combinedData):
    """
    This function can be utilized in the case you only have a single data set and need to split it into both
    a training and testing set

    Args:
        splitRatio (int): The fraction of data (0-1) that will be used for training data
        combinedData (DataFrame): Our initial dataframe prior to splitting

    Returns:
        tuple: Our training and testing dataframes split from the original singular dataframe
    """
    splitIndex = int(splitRatio * len(combinedData))

    trainingData = combinedData.iloc[:splitIndex]
    testData = combinedData.iloc[splitIndex:]

    return trainingData, testData

def targetSplit(trainingData, testData):
    """
    This function splits training and testing dataframes into feature and targets

    Args:
        trainingData (DataFrame):
        testData (DataFrame):

    Returns:
        tuple: Dataframes for both training and testing split into feature and target dataframes
    """
    trainingDataTargets = trainingData['level'].copy()
    trainingDataFeatures = trainingData.drop('level', axis=1)
    testDataTargets = testData['level'].copy()
    testDataFeatures = testData.drop('level', axis=1)
    return trainingDataFeatures, trainingDataTargets, testDataFeatures, testDataTargets

def dataPreprocessing(trainingFile):
    """
    This function utilizes various other dataprocessing functions the ordering of this processing is: Reading the
    data into a pandas DataFrame -> Splitting our dataset into training and testing dataframes ->
    Encoding non-numeric values -> min-max scaling numeric

    values

    Args:
        trainingFile (.csv): .csv file containing our ML-model's training data
    Returns:
        tuple: Returns two DataFrames which have been encoded, imputed, and scaled
    """
    combinedData = fileRead(trainingFile)
    trainingData, testData = trainingSplit(0.8, combinedData)
    trainingData, testData = encoding(trainingData, testData)
    trainingDataFeatures, trainingDataTargets, testDataFeatures, testDataTargets = targetSplit(trainingData, testData)
    scalableFeatureList = trainingDataFeatures.columns
    trainingDataFeatures, testDataFeatures = scaling(trainingDataFeatures, testDataFeatures, scalableFeatureList)
    return trainingDataFeatures, trainingDataTargets, testDataFeatures, testDataTargets

def predictionCombine(prediction, testData):
    """
    This function combines the unused test SurvivorId's and our predicted Survivals into a singular DataFrame

    Args:
        prediction (DataFrame): Our predicted values for survival
        testData (DataFrame): The DataFrame of our testing data
    Returns:
        DataFrame: Final DataFrame with paired survival prediction and PassengerId
    """
    prediction = pd.DataFrame({'level': testData.level, 'Survived': prediction})
    return prediction

