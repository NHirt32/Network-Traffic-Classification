from dataProcessing import dataPreprocessing as dp

trainingData, testData = dp('dataset.csv')

print(trainingData.head())
