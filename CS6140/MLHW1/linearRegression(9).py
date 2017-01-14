import scipy.io
from scipy.linalg import inv
from scipy import array, inner, average, linspace, array_split, eye, dot

dataDict = dict()
file = "C:/Users/Akash/Downloads/linear_regression.mat"
dataDict = scipy.io.loadmat(file)
trnData = dataDict['X_trn']
trnLabels = dataDict['Y_trn']
tstData = dataDict['X_tst']
tstLabels = dataDict['Y_tst']
kFoldSet = [2, 5, 10, len(trnData)]
minRange = 0.001
maxRange = 0.1
intervals = 100
lambdaSet = linspace(minRange, maxRange, intervals)
MAX_VALUE = float('inf')


# finding weightVector using Ridge Regression
def wtVectUsingRR(lam, trnData, trnLabels):
    wtVctr = dot(trnData.transpose(), trnData)
    wtVctr = wtVctr + lam * eye(wtVctr.shape[0])
    wtVctr = dot(inv(wtVctr), trnData.transpose())
    wtVctr = dot(wtVctr, trnLabels)
    return wtVctr

# taking the l2Norm
def l2Norm(data, labels, wtVctr):
    predictedLabels = dot(data, wtVctr)
    changeInL = labels - predictedLabels
    l2N = inner(changeInL.transpose(), changeInL.transpose())[0][0]
    return l2N

# basis
def phi(row, n):
    result = list()
    result.append(1)
    for power in range(1, n + 1):
        for element in row:
            result.append(element ** power)
    return array(result)

# perform the k fold validation
def kFoldValidation(folds, maximumDegree):
    minError = MAX_VALUE
    lambdaFeasible = None
    wVector = None
    td = trnData
    modifiedTrnData = trnDataWithBasis(td, maximumDegree)
    trnDataChunks = array_split(modifiedTrnData, folds)
    trnLabelChunks = array_split(trnLabels, folds)
    for l in lambdaSet:
        hErrorforLam = list()
        for holdOutNumber in range(folds):
            dataForKFoldCrossValidation = kFValidationData(holdOutNumber,
                                                           folds,
                                                           trnDataChunks,
                                                           trnLabelChunks)
            newTrnData, newTrnLabels, holdOutData, holdOutLabels = dataForKFoldCrossValidation
            wtVctr = wtVectUsingRR(l,
                                   newTrnData,
                                   newTrnLabels)
            holdOutError = l2Norm(holdOutData, holdOutLabels, wtVctr)
            hErrorforLam.append(holdOutError)
        hErrorforLam = average(hErrorforLam)
        if minError > hErrorforLam:
            minError = hErrorforLam
            lambdaFeasible = l
            wVector = wtVctr
    print("optimal lambda: " + str(lambdaFeasible))
    print("Training Error = " + str(minError))
    return wVector

# training data with basis function
def trnDataWithBasis(data, maxDegree):
    modifiedData = list()
    for row in data:
        modifiedData.append(phi(row, maxDegree))
    modifiedData = array(modifiedData)
    return modifiedData

# the k fold validation data
def kFValidationData(holdOutNumber, folds, dataChunks, labelChunks):
    holdOutData = dataChunks[holdOutNumber]
    holdOutLabels = labelChunks[holdOutNumber]
    trnDataAfterKF = list()
    trnLabelsAfterKF = list()
    for h in range(folds):
        if h == holdOutNumber:
            continue
        trnDataAfterKF.extend(dataChunks[h])
        trnLabelsAfterKF.extend(labelChunks[h])
    trnDataAfterKF = array(trnDataAfterKF)
    trnLabelsAfterKF = array(trnLabelsAfterKF)
    return (trnDataAfterKF, trnLabelsAfterKF, holdOutData, holdOutLabels)


def trainAndPredict(maximumDegree):
    for foldCount in kFoldSet:
        wtV = kFoldValidation(foldCount, maximumDegree)
        print("Training with " + str(foldCount) +" folds")
        print("Optimal weight vector(transpose) =")
        print(wtV.transpose())
        predict(wtV, maximumDegree)
        print()

def predict(wtV, maximumDegree):
    modifiedTstData = trnDataWithBasis(tstData, maximumDegree)
    tstError = l2Norm(modifiedTstData, tstLabels, wtV)
    print("Testing error = " + str(tstError))

listOfMaximumDegreesOfBasisFunction = [2, 5, 10, 20]
for maximumDegree in listOfMaximumDegreesOfBasisFunction:
    print("Maximum degree of Basis Function = " + str(maximumDegree))
    print()
    trainAndPredict(maximumDegree)
    print()