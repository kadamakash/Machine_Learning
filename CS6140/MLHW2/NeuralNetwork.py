import os
from copy import deepcopy
from numpy import average, array_split, around, dot, array, zeros
from numpy.random import rand
from scipy.io import loadmat
from scipy.special import expit

class NeuralNetwork:
    def __init__(self):

        # loading the input file
        self.dataDict = dict()
        inputData = "C:/Users/Akash/Desktop/MLHW2/data.mat"
        self.dataDict = loadmat(os.path.join(os.getcwd(), inputData))

        self.MAX_VALUE = float('inf')
        self.MIN_VALUE = -float('inf')

        # segregating the  training and testing data sets
        self.trnData = self.dataDict['X_trn']
        self.tstData = self.dataDict['X_tst']

        # segregating the  training and testing label sets
        self.trnLabels = self.dataDict['Y_trn']
        self.tstLabels = self.dataDict['Y_tst']

        # casting the array to specific type
        self.trnLabels = self.trnLabels.astype(int, copy=False)
        self.tstLabels = self.tstLabels.astype(int, copy=False)
        self.trnData = self.trnData.astype(float, copy=False)
        self.tstData = self.tstData.astype(float, copy=False)

        # normalize
        self.nortrnLabels = (self.trnLabels - min(self.trnLabels)) / (max(self.trnLabels) - min(self.trnLabels))
        self.nortstLabels = (self.tstLabels - min(self.tstLabels)) / (max(self.tstLabels) - min(self.tstLabels))

        # total number of features
        self.featuresCount = self.trnData.shape[1]

        # find the unique classes present in the given label
        self.classes = array(self.findClasses())
        self.numberOfClasses = len(self.classes)

        self.learningRate = 0.5

        # set the number of units for each layer
        self.c1 = self.featuresCount
        self.c2 = 2
        self.c3 = 1

        # initializing the layer unit values to zero
        self.inputUnit = zeros((self.c1, 1))
        self.hiddenUnit = zeros((self.c2, 1))
        self.outputUnit = zeros((self.c3, 1))

        # assigining the weight vector for class 1
        self.wv1 = rand(self.c1, self.c2)
        # assigining the weight vector for class 2
        self.wv2 = rand(self.c2, self.c3)

        # assigining the biases for class 1 and 2 as b1 and b2 respectively
        self.b1 = rand(1, self.c2)
        self.b2 = rand(1, self.c3)

        self.foldCount = 5
        self.maxIteration = 1000

    # Function to find the unique classes present in the given labels
    def findClasses(self):
        uniqueClassSet = set()
        for l in self.trnLabels:
            uniqueClassSet.add(l[0])
        for l in self.tstLabels:
            uniqueClassSet.add(l[0])
        return list(sorted(uniqueClassSet))

    # This function does the forward propogation computation
    def feedToFwdProp(self, index):
        # creating the input unit data
        self.inputUnit = array([self.trnData[index].tolist()])
        # computing the summation at hidden unit
        self.hiddenUnit = dot(self.inputUnit, self.wv1) + self.b1
        # applying the sigmoid activation on the summation at hidden layer
        self.hiddenUnit = expit(self.hiddenUnit)
        # computing the summation at output unit
        self.outputUnit = dot(self.hiddenUnit, self.wv2) + self.b2
        # applying the sigmoid activation on the sum at output unit
        self.outputUnit = expit(self.outputUnit)
        # computing the total error for each incoming data
        errorSum = 0.5 * ((self.trnLabels[index] - self.outputUnit) ** 2)
        return sum(sum(errorSum))


    # This function does the backward propogation computation
    def feedToBackProp(self, index):
        delOpLayer = -1 * (self.trnLabels[index] - self.outputUnit) * self.outputUnit * (1 - self.outputUnit)
        derivativeSumErrbyW2 = dot(delOpLayer.transpose(), self.hiddenUnit).transpose()

        # compute the corrected weight vector 1
        correctedW2 = self.wv2 - self.learningRate * derivativeSumErrbyW2

        weightedSumOfDeltasoutputUnit = dot(delOpLayer, self.wv2.transpose())
        deltahiddenUnit = weightedSumOfDeltasoutputUnit * self.hiddenUnit * (1 - self.hiddenUnit)

        # derivative of Error Sum wrt to weight vector 1
        dEtotalBydw1 = dot(deltahiddenUnit.transpose(), self.inputUnit).transpose()

        # compute the corrected weight vector 2
        correctedW1 = self.wv1 - self.learningRate * dEtotalBydw1

        # assign back the corrected weight vector to the original weight vectors
        self.wv1 = correctedW1
        self.wv2 = correctedW2

    # Function to prepare data for k-fold cross validation
    def kfoldData(self, holdOutNumber, folds, dataSplits, labelSplits):
        holdOutData = dataSplits[holdOutNumber]
        holdOutLabels = labelSplits[holdOutNumber]

        trainingDataChunk = list()
        trainingLabelChunk = list()
        # Assemble all of the data except hold out split
        for h in range(folds):
            if h == holdOutNumber:
                continue
            trainingDataChunk.extend(dataSplits[h])
            trainingLabelChunk.extend(labelSplits[h])
        trainingDataChunk = array(trainingDataChunk)
        trainingLabelChunk = array(trainingLabelChunk)
        return (trainingDataChunk, trainingLabelChunk, holdOutData, holdOutLabels)

    # this function performs the k fold cross validation
    def kFoldCrossValidation(self, folds):
        # Split the data and labels equally into chunks, 'folds' in number
        trainingDataSplits = array_split(self.trnData, folds)
        trainingLabelsSplits = array_split(self.nortrnLabels, folds)
        optWV1 = None
        optWV2 = None
        optiB1 = None
        optiB2 = None
        minimumHoldOutError = self.MAX_VALUE
        for holdOutNumber in range(folds):
            dataFromKFold = self.kfoldData(holdOutNumber,
                                           folds,
                                           trainingDataSplits,
                                           trainingLabelsSplits)
            trainingDataChunk, trainingLabelChunk, holdOutData, holdOutLabels = dataFromKFold

            trainingData = trainingDataChunk
            trainingLabels = trainingLabelChunk
            testingData = holdOutData
            testingLabels = holdOutLabels
            inputUnit = zeros((self.c1, 1))
            hiddenUnit = zeros((self.c2, 1))
            outputUnit = zeros((self.c3, 1))
            w1 = rand(self.c1, self.c2)
            w2 = rand(self.c2, self.c3)
            b1 = rand(1, self.c2)
            b2 = rand(1, self.c3)
            w1, w2, b1, b2, trainingError = self.train(trainingData, trainingLabels, inputUnit, outputUnit,
                                                       hiddenUnit, w1, w2, b1, b2)
            predictions, holdOutError = self.predict(testingData, testingLabels, w1, w2, b1, b2)
            # take the weight vectors and biases if whose error in prediction is minimim
            if holdOutError < minimumHoldOutError:
                optWV1 = w1
                optWV2 = w2
                optiB1 = b1
                optiB2 = b2
        return (optWV1, optWV2, optiB1, optiB2)

    def train(self, trnData, trnLabels, inputUnit, outputUnit, hiddenUnit, wv1, wv2, b1, b2):
        for i in range(self.maxIteration):
            error = 0
            for j in range(len(trnData)):
                # Forward propogation
                inputUnit = array([trnData[j].tolist()])
                hiddenUnit = dot(inputUnit, wv1) + b1
                hiddenUnit = expit(hiddenUnit)
                outputUnit = dot(hiddenUnit, wv2) + b2
                outputUnit = expit(outputUnit)
                errorTotal = 0.5 * ((trnLabels[j] - outputUnit) ** 2)
                error += sum(sum(errorTotal))
                
                # Back Prapagation
                delOpLayer = -1 * (trnLabels[j] - outputUnit) * outputUnit * (1 - outputUnit)
                # taking derivative of Error Total w.r.t weight vector 2
                dEtotalBydw2 = dot(delOpLayer.transpose(), hiddenUnit).transpose()
                # compute the corrected weight vector 2
                correctedWV2 = wv2 - self.learningRate * dEtotalBydw2
                weightedSumOfDeltasoutputUnit = dot(delOpLayer, wv2.transpose())
                deltahiddenUnit = weightedSumOfDeltasoutputUnit * hiddenUnit * (1 - hiddenUnit)
                # taking derivative of ErrorTotal w.r.t dw1
                dEtotalBydw1 = dot(deltahiddenUnit.transpose(), inputUnit).transpose()
                # get the corrected weight vector
                correctedWV1 = wv1 - self.learningRate * dEtotalBydw1
                
                # assigin back the corrected weight vectors
                wv1 = correctedWV1
                wv2 = correctedWV2
        return (wv1, wv2, b1, b2, error)


    # check the prediction for test data once you are done training your network
    def test(self):
        optiWV1, optiWV2, optiB1, optiB2 = self.kFoldCrossValidation(self.foldCount)
        predictions, error = self.predict(self.tstData, self.nortstLabels, optiWV1, optiWV2, optiB1, optiB2)
        # predictions = around(array(list(map(lambda a : a[0][0], predictions))) * 2).tolist()
        print("Testing accuracy:" + str(self.calculateAccuracy(predictions, self.tstLabels))+"%")


    def calculateAccuracy(self, predictions, labels):
        predictions = around(array(list(map(lambda p : p[0][0], predictions))) * 2)
        numberOfErrors = 0
        for i in predictions - labels.transpose()[0]:
            if 0 != i:
                numberOfErrors += 1
        return 100 * (len(predictions) - numberOfErrors) / len(predictions)

    def mainTest(self):
        nodeCountList = [10, 20, 30, 50, 100]
        for nodeCount in nodeCountList:
            print("Hidden Layer Unit Count= " + str(nodeCount))
            self.c2 = nodeCount
            self.test()
            print()


    # apply the test data to the above trained network which has the updated weight vector
    def predict(self, tstData, tstLabels, wv1, wv2, b1, b2):
        predictions = list()
        error = 0
        for i in range(len(tstData)):
            inputUnit = deepcopy(array([tstData[i].tolist()]))
            hiddenUnit = dot(inputUnit, wv1) + b1
            hiddenUnit = expit(hiddenUnit)
            outputUnit = dot(hiddenUnit, wv2) + b2
            outputUnit = expit(outputUnit)
            predictions.append(outputUnit)
            error += sum(sum(0.5 * ((tstLabels[i] - outputUnit) ** 2)))
        return (predictions, error)

if __name__ == "__main__":
    obj = NeuralNetwork()
    obj.mainTest()


