from scipy.io import loadmat
from scipy import dot, average, zeros, log
from copy import deepcopy
from scipy.special import expit

data = dict()
input = "C:/Users/Akash/Downloads/logistic_regression.mat"
data = loadmat(input)
trnData = data['X_trn']
tstData = data['X_tst']
trnLabels = data['Y_trn']
tstLabels = data['Y_tst']
MAXIMUM_VALUE = float('inf')
MINIMUM_VALUE = -float('inf')

def newClasses():
    labelSet = set()
    for eachlabel in trnLabels:
        labelSet.add(eachlabel[0])
    for eachlabel in tstLabels:
        labelSet.add(eachlabel[0])
    return list(sorted(labelSet))

# creating class label
def createClassLabel():
    trainingLabelsforClass = list()
    testingLabelsforClass = list()
    for c in classes:
        newCpy = deepcopy(trnLabels)
        trainingLabelsforClass.append(newCpy)
        newtesting = deepcopy(tstLabels)
        testingLabelsforClass.append(newtesting)
    for c in classes:
        for i in range(trnLabels.shape[0]):
            if trnLabels[i][0] == c:
                trainingLabelsforClass[c][i][0] = 1
            else:
                trainingLabelsforClass[c][i][0] = 0
        for i in range(tstLabels.shape[0]):
            if tstLabels[i][0] == c:
                testingLabelsforClass[c][i][0] = 1
            else:
                testingLabelsforClass[c][i][0] = 0

    return (trainingLabelsforClass, testingLabelsforClass)

def wtVectForClass():
    wtVectList = list()
    for c in classes:
        wtVectList.append(zeros((1, Features)))
    return wtVectList


classes = newClasses()
trainingLabelsforClass = None
testingLabelsforClass = None
trainingLabelsforClass, testingLabelsforClass = createClassLabel()
TrainingSamples = trnData.shape[0]
Features = trnData.shape[1]
Classes = len(classes)
wtVectForClasses = wtVectForClass()

# compute cost
def cost(wtVect, data, inputlabels):
    exp1 = -1 * inputlabels * log(expit(dot(data, wtVect.transpose())))
    exp2 = (1 - inputlabels) * log(1 - expit(dot(data, wtVect.transpose())))
    return average(exp1 - exp2)

# compute the gradient
def computeGradient(wtVect, data, inputlabels):
    gradientVector = zeros((1, Features))
    error = expit(dot(data, wtVect.transpose())) - inputlabels
    for f in range(Features):
        col = zeros((data.shape[0], 1))
        for j in range(data.shape[0]):
            col[j][0] = data[j][f]
            gradientVector[0][f] = average(error * col)
    return gradientVector

# training the data
def training():
    alpha = 0.5
    iteration = 400
    tolerance = 1e-8
    for item in classes:
        print("Weight Vector Class " + str(item))
        c = iteration
        convBool = False
        while 0 < c and not convBool:
            gradValue = computeGradient(wtVectForClasses[item],
                                        trnData,
                                        trainingLabelsforClass[item])
            oldCost = cost(wtVectForClasses[item],
                           trnData,
                           trainingLabelsforClass[item])
            wtVectForClasses[item] -= alpha * gradValue
            newCost = cost(wtVectForClasses[item],
                           trnData,
                           trainingLabelsforClass[item])
            c -= 1
            convBool = abs(newCost - oldCost) < tolerance
        print(wtVectForClasses[item])
        print()
    return

# predicting
def predict(data, labels):
    predict= list()
    for item in classes:
        predict.append(dot(data, wtVectForClasses[item].transpose()))
    predictedLabels = deepcopy(labels)


    for i in range(len(labels)):
        predictedValue = None
        maxProbability = MINIMUM_VALUE
        for eachClass in classes:
            if predict[eachClass][i][0] > maxProbability:
                maxProbability = predict[eachClass][i][0]
                predictedValue = eachClass
        predictedLabels[i][0] = predictedValue

    fError = list()
    c = 0
    totalC = 0
    for i in range(predictedLabels.shape[0]):
        if predictedLabels[i][0] == labels[i][0]:
            fError.append(1)
        else:
            fError.append(0)
            c += 1
        totalC +=1
    print("Error " + str(c) + " from " + str(totalC))
    print("Prediction Accuracy = " +str(100 * (average(fError))))


training()

print ("Training Data Report")
predict(trnData, trnLabels)
print()

print ("Testing Data Report")
predict(tstData, tstLabels)