import os
from copy import deepcopy
from scipy.io import loadmat
from matplotlib import pyplot as plotter
from scipy import mean, rand, array,  exp, zeros, ones
from numpy.linalg import eig, norm
from sklearn.preprocessing import MinMaxScaler


class SC:
    def __init__(self):
        self.dataDict = dict()
        self.MAX_VALUE = float('inf')
        self.MIN_VALUE = -float('inf')

        inputPath = "C:/Users/Akash/Desktop/CS6140/MLHW3/data.mat"
        self.dataDict = loadmat(os.path.join(os.getcwd(), inputPath))
        self.trainingData = self.dataDict['X_Question2_3']

    def spectral(self, k, sigma):
        wtMat = self.calculateWeightMatrix(sigma)
        degreeMat = self.calculateDegreeMatrix(wtMat)

        laplacian = degreeMat - wtMat
        laplacian = MinMaxScaler().fit_transform(laplacian)

        # compute the eigen values and eigen vectors
        eVal, eVect = eig(laplacian)
        eVect = eVect.real
        eVect = MinMaxScaler().fit_transform(eVect)
        sortedIndices = eVal.argsort()

        # get the bottom k indices
        bottomK = sortedIndices[:k]

        H = eVect[:, bottomK]
        HTranspose = H.transpose()
        centroidsToDataPointIndices = self.kMean(k, 15, HTranspose)

        x_limits = [min(self.trainingData[0, :]) - 2, max(self.trainingData[0, :]) + 2]
        y_limits = [min(self.trainingData[1, :]) - 2, max(self.trainingData[1, :]) + 2]
        self.clusteredDataPlot(centroidsToDataPointIndices, x_limits, y_limits)


    def kMean(self, k, rnd, data):
        print(data.shape)
        minimumCost = self.MAX_VALUE

        optimumCentroids = None
        for indx in range(rnd):
            centroids = rand(data.shape[0], k)
            centroidsToDataPoints = dict()
            centroidsToDataPointIndices = dict()
            maxItr = 1000

            for i in range(maxItr):
                centroidsToDataPoints = dict()
                centroidsToDataPointIndices = dict()
                clusters = ones(data.shape[1]) * -1

                # Update the centroids for data points
                for dpIdx in range(data.shape[1]):
                    dataPoint = data[:, dpIdx]
                    closestCentroidIdx = -1
                    closestCentroidEDist = self.MAX_VALUE

                    for cIndx in range(centroids.shape[1]):
                        centroid = centroids[:, cIndx]
                        distSq = 2 ** norm(dataPoint - centroid)
                        if distSq < closestCentroidEDist:
                            closestCentroidEDist = distSq
                            closestCentroidIdx = cIndx
                    clusters[dpIdx] = closestCentroidIdx
                    if closestCentroidIdx not in centroidsToDataPoints:
                        centroidsToDataPoints[closestCentroidIdx] = list()
                    if closestCentroidIdx not in centroidsToDataPointIndices:
                        centroidsToDataPointIndices[closestCentroidIdx] = list()
                    centroidsToDataPoints[closestCentroidIdx].append(dataPoint)
                    centroidsToDataPointIndices[closestCentroidIdx].append(dpIdx)

                # Update the centroids based on data points
                newCentroids = deepcopy(centroids)
                for cIndx in range(centroids.shape[1]):
                    if cIndx in centroidsToDataPoints:
                        newCentroids[:, cIndx] = mean(array(centroidsToDataPoints[cIndx]).transpose(), 1)
                if (centroids != newCentroids).all():
                    centroids = newCentroids
                else:
                    break
            cost = self.calculateCost(centroids, centroidsToDataPoints)
            if minimumCost > cost:
                # bestCentroids = centroids
                # bestCentroidsToDataPoints = centroidsToDataPoints
                optimumCentroids = centroidsToDataPointIndices
        return optimumCentroids

    def calculateDegreeMatrix(self, wtMat):
        degreeMat = zeros(shape=(self.trainingData.shape[1], self.trainingData.shape[1]), dtype=float)
        for i in range(wtMat.shape[0]):
            degreeMat[i][i] = sum(wtMat[i, :])
        return degreeMat

    def calculateCost(self, centroids, centroidsToDataPoints):
        numberOfDataPoints = 0
        totalOfdistSq = 0
        for cIndx in centroidsToDataPoints:
            centroid = centroids[:, cIndx]
            for datapoint in centroidsToDataPoints[cIndx]:
                numberOfDataPoints += 1
                totalOfdistSq += norm(centroid - datapoint) ** 2
        cost = totalOfdistSq / numberOfDataPoints
        return cost

    def calculateWeightMatrix(self, sigma):
        wtMat = ones(shape=(self.trainingData.shape[1], self.trainingData.shape[1]), dtype=float)
        for i in range(wtMat.shape[0]):
            for j in range(wtMat.shape[1]):
                wtMat[i][j] = exp(-1 * (norm(self.trainingData[:, i] - self.trainingData[:, j]) ** 2) / sigma)
        return wtMat

    # plotting the clustered data
    def clusteredDataPlot(self, centroidsToDataPointIndices, x_limits, y_limits):
        centroidsToDataPoints = dict()
        print("Centroid data point index", centroidsToDataPointIndices)
        for cIndx in centroidsToDataPointIndices:
            centroidsToDataPoints[cIndx] = list()
            for dpIdx in centroidsToDataPointIndices[cIndx]:
                centroidsToDataPoints[cIndx].append(self.trainingData[:, dpIdx])

        for cIndx in centroidsToDataPoints:
            dataPointsForCentroid = array(centroidsToDataPoints[cIndx]).transpose()
            plotter.plot(dataPointsForCentroid[0, :], dataPointsForCentroid[1, :], 'o')
        plotter.xlim(x_limits)
        plotter.ylim(y_limits)
        # plotter.savefig(os.path.join(os.getcwd(), 'clustered_graph.png'))
        plotter.show()


spectral = SC()
spectral.spectral(4, 1)