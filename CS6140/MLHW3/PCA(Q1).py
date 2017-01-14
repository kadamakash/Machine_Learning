from scipy.io import loadmat
import numpy as np
import numpy.linalg as e
from matplotlib import pyplot as plotter
import os

inputData = dict()
inputPath = "C:/Users/Akash/Desktop/CS6140/MLHW3/data.mat"

inputData = loadmat(inputPath)
pcaData = inputData['X_Question1']

# compute D(40) dimesional mean vector
meanVector = pcaData.mean(1)
print("MeanVector")
print(meanVector)
# compute covariance matrix for whole data set
for r in range(pcaData.shape[0]):
    for c in range(pcaData.shape[1]):
        pcaData[r][c] -= meanVector[r]

#print(np.dot(np.array([pcaData[1].transpose().tolist()]).transpose(), np.array([pcaData[1].transpose().tolist()])))
covmat = np.cov(pcaData)

# compute eigen vectors and corresponding eigen values
eigvals, eigvecs = e.eig(covmat)

# sort the eigen vectors by decreasing order of eigen values and
sortedInd = eigvals.argsort()[::-1]
eigvals = eigvals[sortedInd]
eigvecs = eigvecs[:, sortedInd]

# choose 2 eigen vectors with the largest eigenvalues to form a 40 * 2 dimensional matrix
basis = eigvecs[:, [0,1]]
print()
print("Basis")
print(basis)

# Use the eigen vector matrix to transform the samples onto the new subspace
y = np.dot(basis.transpose(), pcaData)
print()
print("2 dimensional representation of the data")
print(y)

xRange = [min(y[0, :]) - 2, max(y[0, :]) + 2]
yRange = [min(y[1, :]) - 2, max(y[1, :]) + 2]
plotter.plot(y[0, :], y[1, :], 'o')
plotter.xlabel("dimension 1")
plotter.ylabel("dimension 2")
plotter.savefig(os.path.join(os.getcwd(), 'PCAPlot.png'))






