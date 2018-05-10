# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 16:36:08 2018

@author: Cameron Hargreaves
"""

import numpy as np
import csv
import math
import matplotlib.pyplot as plt

def main():
    fileNames = ["animals", "countries", "fruits", "veggies"]
    numEpochs = 5
    numIterations = 50
    
    scores = []
    
    for epoch in range(numEpochs):
        precisionClassifications = []
        recallClassifications = []
        fMeasureClassifications = []
        
        inputData = lTwoNormalize(readData())
        labels = readLabels(fileNames)
            
        for numClusters in range(1,11):
            centroids = np.empty((numClusters, len(inputData[0])))
            for i in range(numClusters):
                centroids[i] = generateCentroid(inputData)
                
            for i in range(numIterations):
                classifications = classifyOnCentroid(centroids, inputData)
                centroids = updateCentroids(classifications, inputData, numClusters)
                print("Epoch: " + str(epoch) + " K: " + str(numClusters) + " Iteration: " + str(i))
                
            contingencyTableClassifications = contingencyTable(labels, classifications)
            
            precisionClassifications.append(precision(contingencyTableClassifications))
            recallClassifications.append(recall(contingencyTableClassifications))
            fMeasureClassifications.append(fMeasure(contingencyTableClassifications))
            
            print("Precision: " + str(precisionClassifications[numClusters - 1]) + 
                  "\nRecall: " + str(recallClassifications[numClusters - 1]) +
                  "\nF-Measure: " + str(fMeasureClassifications[numClusters - 1])) 
        
        scores.append(np.asarray([precisionClassifications, recallClassifications, fMeasureClassifications]))
    
    summedScore = np.zeros(scores[0].shape)
    
    for score in scores:
        summedScore = np.add(summedScore, score)
    averageScore = np.divide(summedScore, 10)
    
    plt.plot(range(1,11), averageScore[0], label="Precision")
    plt.plot(range(1,11), averageScore[1], label="Recall")    
    plt.plot(range(1,11), averageScore[2], label="F-Measure")    
    plt.legend()
    plt.show
    
def cosineSimilarity(centrePoint, row):
    vectorMult = 0
    for i in range(len(centrePoint)):
        vectorMult += centrePoint[i] * row[i]
    cosSim = vectorMult / (lTwoNorm(centrePoint) * lTwoNorm(row))
    return cosSim
    
def manhattanDist(centrePoint, row):
    sumDiff = 0
    for i in range(len(centrePoint)):
        sumDiff += abs(centrePoint[i] - row[i])
    return sumDiff
    
def lTwoNormalize(inputData):
    normalizedData = np.zeros(inputData.shape)
    for i, row in enumerate(inputData):
        norm = lTwoNorm(row)
        normalizedData[i] = np.divide(inputData[i], norm)
    return normalizedData    
        
def lTwoNorm(vector):
    summedRow = 0
    for value in vector:
        summedRow += value ** 2
    norm = math.sqrt(summedRow)
    return norm

def contingencyTable(labels, classifications):
    truePositive = falsePositive = trueNegative = falseNegative = 0
    for i in range(len(labels)):
        for j in range(i, len(labels)):
            if labels[i][1] == labels[j][1] and classifications[i] == classifications[j]:
                truePositive += 1
            elif labels[i][1] != labels[j][1] and classifications[i] == classifications[j]:
                falsePositive += 1
            elif labels[i][1] == labels[j][1] and classifications[i] != classifications[j]:
                falseNegative += 1
            elif labels[i][1] == labels[j][1] and classifications[i] != classifications[j]:
                trueNegative += 1
    return [truePositive, falsePositive, falseNegative, trueNegative]

def precision(contingencyTable):
    return contingencyTable[0] / (contingencyTable[0] + contingencyTable[1])

def recall(contingencyTable):
    return contingencyTable[0] / (contingencyTable[0] + contingencyTable[2])

def fMeasure(contingencyTable):
    return (2 * precision(contingencyTable) * recall(contingencyTable)) / (precision(contingencyTable) + recall(contingencyTable))

def readData():
    animals = np.genfromtxt("clustering-data/animals", delimiter = " ")
    countries = np.genfromtxt("clustering-data/countries", delimiter = " ")
    fruits = np.genfromtxt("clustering-data/fruits", delimiter = " ")
    veggies = np.genfromtxt("clustering-data/veggies", delimiter = " ")
    
    inputData = np.concatenate((animals, countries, fruits, veggies))
    inputData = np.delete(inputData, 0, 1)  # Remove the labels 
    
    return inputData

def readLabels(fileNames):
    labels = []   # Create four lists for the labels
    for i, fileName in enumerate(fileNames):
        with open("clustering-data/" + str(fileNames[i]), "r") as inputFile:
            reader = csv.reader(inputFile, delimiter = ' ')
            for row in reader:
                labels.append([row[0], fileName])
    return labels

def euclidDist(centrePoint, row):
    sumDiff = 0
    for i in range(len(centrePoint)):
        sumDiff += (centrePoint[i] - row[i])**2
    return math.sqrt(sumDiff)
    
def generateCentroid(inputData):
    maxValues = np.amax(inputData, axis=0)
    minValues = np.amin(inputData, axis=0)
    centroid = []
    for i in range(len(maxValues)):
        centroid.append(np.random.uniform(low = minValues[i], high = maxValues[i]))
    return np.asarray(centroid)

def nearestCentroid(centroids, inputRow):
    distances = []
    for centroid in centroids:
        distances.append(cosineSimilarity(centroid, inputRow)) # Calc each euclid distance
    return distances.index(min(distances))  # return index of nearest centroid distance
    
def classifyOnCentroid(centroids, inputData):
    classifications = []
    for row in inputData:
        classifications.append(nearestCentroid(centroids, row))
    return classifications    
    
def updateCentroids(classifications, inputData, numClusters):
    centroids = np.empty((numClusters, len(inputData[0])))
    for i, classification in enumerate(classifications):        # Sum all the classified points
        centroids[classification] = np.add(centroids[classification], inputData[i])
    centroids = np.divide(centroids, len(inputData))
    return centroids
    
main()