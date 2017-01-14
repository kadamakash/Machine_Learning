import os
from numpy import dot, inner, average, zeros
from copy import deepcopy
from random import sample
from scipy.io import loadmat
from scipy.special import expit


class SVM:
    def __init__(self):

        # loading the input file
        self.dataDict = dict()
        inputFile = "C:/Users/Akash/Desktop/MLHW2/data.mat"
        self.dataDict = loadmat(os.path.join(os.getcwd(), inputFile))

        self.MAX_VALUE = float('inf')
        self.MIN_VALUE = -float('inf')

        # segregating the  training and testing data sets
        self.trnData = self.dataDict['X_trn']
        self.tstData = self.dataDict['X_tst']

        # segregating the  training and testing label sets
        self.trnLabels = self.dataDict['Y_trn']
        self.tstLabels = self.dataDict['Y_tst']

        # casting the array to specific type
        self.trnData = self.trnData.astype(float, copy=False)
        self.tstData = self.tstData.astype(float, copy=False)
        self.trnLabels = self.trnLabels.astype(int, copy=False)
        self.tstLabels = self.tstLabels.astype(int, copy=False)

        # find the unique classes present in the given label
        self.classes = self.findClasses()
        self.trainingLabelsForClass = None
        self.testingLabelsForClass = None
        self.createLabelForClass()

        self.trnSampleCount = self.trnData.shape[0]
        self.featuresCount = self.trnData.shape[1]
        self.classCount = len(self.classes)

    # Function to find the unique classes present in the given labels
    def findClasses(self):
        uniqueClassSet = set()
        for lbl in self.trnLabels:
            uniqueClassSet.add(lbl[0])
        for lbl in self.tstLabels:
            uniqueClassSet.add(lbl[0])
        return list(sorted(uniqueClassSet))

    # This function creates the labels for class and assigns 1 if the label
    # belongs to the class else it is assigned -1. This is to differentiate a class from rest of the classes
    def createLabelForClass(self):
        self.trainingLabelsForClass = list()
        self.testingLabelsForClass = list()
        # creating copies of the labels
        for cls in self.classes:
            cpy = deepcopy(self.trnLabels)
            self.trainingLabelsForClass.append(cpy)
            cpy = deepcopy(self.tstLabels)
            self.testingLabelsForClass.append(cpy)
        # Differentiate each class from the rest of the classes
        for cls in self.classes:
            for i in range(self.trnLabels.shape[0]):
                if cls == self.trnLabels[i]:
                    self.trainingLabelsForClass[cls][i] = 1
                else:
                    self.trainingLabelsForClass[cls][i] = -1
            for i in range(self.tstLabels.shape[0]):
                if cls == self.tstLabels[i]:
                    self.testingLabelsForClass[cls][i] = 1
                else:
                    self.testingLabelsForClass[cls][i] = -1

    # computing using (2) from the paper
    def E(self, i, data, labels, alphas, b):
        return self.func(data, b, data[i], labels, alphas) - labels[i]

    def func(self, data, b, dataPoint, labels, alphas):
        op = 0
        for i in range(len(data)):
            op += alphas[i] * labels[i] * inner(data[i], dataPoint)
        op += b
        return op

    def computeB(self, b, i, j, Ei, Ej, C, alphas, prevAlphaI, prevAlphaJ, data, labels):
        t1 = b
        t2 = Ei
        t3 = labels[i] * (alphas[i] - prevAlphaI) * inner(data[i], data[i])
        t4 = labels[j] * (alphas[j] - prevAlphaJ) * inner(data[i], data[j])

        b1 = t1 - t2 - t3 - t4

        t2 = Ej
        t3 = labels[i] * (alphas[i] - prevAlphaI) * inner(data[i], data[j])
        t4 = labels[j] * (alphas[j] - prevAlphaJ) * inner(data[j], data[j])
        b2 = t1 - t2 - t3 - t4

        if ((0 < alphas[i]) and (alphas[i] < C)) and not ((0 < alphas[j]) and (alphas[j] < C)):
            return b1
        elif ((0 < alphas[j]) and (alphas[j] < C)) and not ((0 < alphas[i]) and (alphas[i] < C)):
            return b2
        else:
            return (b1 + b2) / 2

    def computeLH(self, i, j, labels, alphas, C):
        L = self.MIN_VALUE
        H = self.MAX_VALUE
        if (labels[i] != labels[j]):
            L = max(0, alphas[j] - alphas[i])
            H = min(C, C + alphas[j] - alphas[i])
        else:
            L = max(0, alphas[i] + alphas[j] - C)
            H = min(C, alphas[i] + alphas[j])
        return (L, H)

    # compute Eta using (14) in paper
    def computeEta(self, i, j, data):
        op = 2 * inner(data[i], data[j]) - inner(data[i], data[i]) - inner(data[j], data[j])
        return op

    # computing new alpha j using (12) and (15) form paper
    def findNewAJ(self, i, j, alphas, data, labels, Ei, Ej, L, H):
        eta = self.computeEta(i, j, data)
        newAJ = alphas[j] - (labels[j] * (Ei - Ej) / eta)
        if newAJ > H:
            return H
        elif newAJ < L:
            return L
        else:
            return newAJ

    # compute the weight vector for the specific class
    def findWV(self, alphas, inputLabels, data):
        # get the labels from the inputLabels
        labels = inputLabels.transpose().tolist()[0]
        w = alphas[0] * labels[0] * data[0]
        for i in range(1, len(alphas)):
            w += alphas[i] * labels[i] * data[i]
        return w

    def performSmo(self, inData, inputLabels):
        data = inData

        # lagrange multipliers for solution and initializing all alpha to 0
        labels = inputLabels.transpose().tolist()[0]
        alphas = zeros(inputLabels.shape)
        alphas = alphas.transpose().tolist()[0]

        # threshold for solution, initializing to 0
        b = 0
        passes = 0
        MAX_PASSES = 2
        C = 1
        tolerance = 0.01
        trnSamplesIndx = list(range(self.trnSampleCount))

        while passes < MAX_PASSES:

            numberOfChangedAlphas = 0

            for i in trnSamplesIndx:
                Ei = self.E(i, data, labels, alphas, b)
                if ((labels[i] < -1 * tolerance) and (alphas[i] < C)) or (
                            (labels[i] > -1 * tolerance) and (alphas[i] > 0)):
                    # selecting j randomly and j != i
                    j = sample(trnSamplesIndx[:i] + trnSamplesIndx[i + 1:], 1)[0]

                    # Compute Ej using (2) in paper
                    Ej = self.E(j, data, labels, alphas, b)

                    # saving old alphas
                    prevAlphaI = alphas[i]
                    prevAlphaJ = alphas[j]

                    L, H = self.computeLH(i, j, labels, alphas, C)

                    if L == H:
                        continue
                    eta = self.computeEta(i, j, data)
                    if eta >= 0:
                        continue
                    # computing alpha j using (12) and (15) form paper
                    alphas[j] = self.findNewAJ(i, j, alphas, data, labels, Ei, Ej, L, H)

                    if abs(alphas[j] - prevAlphaJ) < 0.00001:
                        continue
                    alphas[i] = alphas[i] + labels[i] * labels[j] * (prevAlphaJ - alphas[j])

                    # compute b using (19) from paper
                    b = self.computeB(b, i, j, Ei, Ej, C, alphas, prevAlphaI, prevAlphaJ, data, labels)
                    numberOfChangedAlphas += 1
            if numberOfChangedAlphas == 0:
                passes += 1
            else:
                passes = 0

        return (alphas, b)

    def softMaxforSMO(self):
        # list of all the weight vectors for a specific class
        weigthVforClass = list()
        # bias for class
        classwiseB = list()
        # predictions for classes
        classPredictions = list()

        for i in range(len(self.classes)):
            labels = self.trainingLabelsForClass[i]
            alphas, b = self.performSmo(self.trnData, labels)

            wtv = self.findWV(alphas, labels, self.trnData)

            # weight vector for class
            weigthVforClass.append(wtv)

            # bias for class
            classwiseB.append(b)

            # prediction for class
            predictions = expit(dot(self.tstData, wtv.transpose()) + b)
            classPredictions.append(predictions)

        actualPredictions = list()

        # computing the predictions
        for i in range(len(self.tstLabels)):
            predictedClass = None
            maxprob = self.MIN_VALUE
            for cIdx in range(len(self.classes)):
                if classPredictions[cIdx][i] > maxprob:
                    maxprob = classPredictions[cIdx][i]
                    predictedClass = self.classes[cIdx]
            actualPredictions.append(predictedClass)

        # print weight vector for class
        print("weigth Vectors for Class: " + str(weigthVforClass))
        # print bias for class
        print("classwise Biases: " + str(weigthVforClass))
        # print the actual prediction
        print("Predictions: " + str(actualPredictions))
        # print testing labels for class
        print("Testing labels: " + str(self.tstLabels.transpose().tolist()))

if __name__ == "__main__":
    obj = SVM()
    obj.softMaxforSMO()
