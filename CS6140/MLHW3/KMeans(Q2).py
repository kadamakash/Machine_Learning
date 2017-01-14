from copy import deepcopy
from scipy.linalg import norm
from scipy.io import loadmat
from scipy import array, mean, rand, ones
import os
from matplotlib import pyplot as plotter

# this is to compute the cost function
def computeCostFunc(centroids, centroidsToDataPoints):
    numberOfDataPoints = 0
    totalOfdistSqrd = 0

    for cIndex in centroidsToDataPoints:
        ctr = centroids[:, cIndex]
        for datapoint in centroidsToDataPoints[cIndex]:
            numberOfDataPoints += 1
            totalOfdistSqrd += norm(ctr - datapoint) ** 2
    cost = totalOfdistSqrd / numberOfDataPoints
    return cost


def KMeansClustering(trainingData, cluster, rnd):

    minCost = MAX_VALUE
    optimumCentersToDataPoints = None

    for i in range(rnd):
        oldCenters = rand(trainingData.shape[0], cluster)
        centroidsToDataPoints = dict()
        centroidsToDataPointsIndex = dict()
        maxIterations = 1000

        for i in range(maxIterations):
            centroidsToDataPoints = dict()
            centroidsToDataPointsIndex = dict()
            newClusters = ones(trainingData.shape[1]) * -1

            # Compute the centroid for each data point
            for dpIdx in range(trainingData.shape[1]):
                dataPnt = trainingData[:, dpIdx]
                closestCenterIdx = -1
                closestCenterDistSqrd = MAX_VALUE

                for centroidIndex in range(oldCenters.shape[1]):
                    eachCentroid = oldCenters[:, centroidIndex]
                    distanceSquared = 2 ** norm(dataPnt - eachCentroid)
                    if closestCenterDistSqrd > distanceSquared:
                        closestCenterDistSqrd = distanceSquared
                        closestCenterIdx = centroidIndex
                newClusters[dpIdx] = closestCenterIdx


                if closestCenterIdx not in centroidsToDataPoints:
                    centroidsToDataPoints[closestCenterIdx] = list()
                if closestCenterIdx not in centroidsToDataPointsIndex:
                    centroidsToDataPointsIndex[closestCenterIdx] = list()
                centroidsToDataPoints[closestCenterIdx].append(dataPnt)
                centroidsToDataPointsIndex[closestCenterIdx].append(dpIdx)

            # Update the centroids based on data points
            newCentroids = deepcopy(oldCenters)
            for centroidIndex in range(oldCenters.shape[1]):
                if centroidIndex in centroidsToDataPoints:
                    newCentroids[:, centroidIndex] = mean(array(centroidsToDataPoints[centroidIndex]).transpose(), 1)
            if (oldCenters == newCentroids).all():
                break
            else:
                oldCenters = newCentroids
        cost = computeCostFunc(oldCenters, centroidsToDataPoints)

        # Choose the best Cluster based on Cost function
        if minCost > cost:
            optimumCentersToDataPoints = centroidsToDataPointsIndex
    return optimumCentersToDataPoints

# clustering of the data (indices of points in each group) for the best run of the kmeans
def clusteredData(centroidsToDataPoints):
    result = []
    for cluster in centroidsToDataPoints:
        for i in centroidsToDataPoints[cluster]:
            result.append((cluster, i))
    result = list(sorted(result, key =lambda x:x[1]))
    result = list(map (lambda x: x[0], result))
    return result

# Plotting Original Graph
def plotOriginalGraph(trainingData, xLimit, yLimit):
    plotter.plot(trainingData[0, :], trainingData[1, :], 'o')
    plotter.xlim(xLimit)
    plotter.ylim(yLimit)
    plotter.savefig(os.path.join(os.getcwd(), '2OriginalDataGraph.png'))

# Plotting Graph after Kmeans Clustering
def clusteredDataPlot(centroidsToDataPoints, trainingData, xLimit, yLimit):
    dpToDict = dict()

    for cIndex in centroidsToDataPoints:
        dpToDict[cIndex] = list()
        for i in centroidsToDataPoints[cIndex]:
            dpToDict[cIndex].append(trainingData[:, i])

    for cIndex in dpToDict:
        centroidData = array(dpToDict[cIndex]).transpose()
        plotter.plot(centroidData[0, :], centroidData[1, :], 'o')
    plotter.xlim(xLimit)
    plotter.ylim(yLimit)
    plotter.savefig(os.path.join(os.getcwd(), '2ClusteredDataGraph.png'))


def kmeans(trainingData, numOfCluster, randomIteration):
    centroidsToDataPointsIndices = KMeansClustering(trainingData, numOfCluster, randomIteration)
    result = clusteredData(centroidsToDataPointsIndices)

    print ("Indices of points in each group")
    print(result)

    xLimit = [min(trainingData[0, :]) - 2, max(trainingData[0, :]) + 2]
    yLimit = [min(trainingData[1, :]) - 2, max(trainingData[1, :]) + 2]
    plotOriginalGraph(trainingData, xLimit, yLimit)
    clusteredDataPlot(centroidsToDataPointsIndices, trainingData, xLimit, yLimit)


inputPath = "C:/Users/Akash/Desktop/CS6140/MLHW3/data.mat"
data = dict()
dataDict = loadmat(os.path.join(os.getcwd(), inputPath))
trainingData = dataDict['X_Question2_3']

MAX_VALUE = float('inf')
MIN_VALUE = -float('inf')

kmeans(trainingData, 4, 5)