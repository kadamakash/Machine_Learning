from numpy import array
from scipy.io import loadmat
from sklearn import svm

dataDict = dict()
dataDict = loadmat("C:/Users/Akash/Desktop/MLHW2/data.mat")
trnData = dataDict['X_trn']
tstData = dataDict['X_tst']
trnLabels = dataDict['Y_trn']
tstLabels = dataDict['Y_tst']

trnData = trnData.astype(float, copy=False)
tstData = tstData.astype(float, copy=False)
trnLabels = trnLabels.astype(int, copy=False)
tstLabels = tstLabels.astype(int, copy=False)

kernelList = ['linear', 'poly', 'rbf', 'sigmoid']

for i in kernelList:
    clf = svm.SVC(kernel=i)
    fitting = clf.fit(trnData, array(trnLabels.transpose().tolist()[0]))
    prediction = clf.predict(tstData)

    print(fitting)
    print(fitting._get_coef())
    print(prediction)
    print()



