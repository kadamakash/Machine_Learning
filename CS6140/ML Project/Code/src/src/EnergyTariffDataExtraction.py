import os
import csv
import re

class EnergyTariffDataExtractor:
    def __init__(self):
        self.file = os.path.join(os.path.dirname(os.getcwd()), 'EnergyData', 'Average_retail_price_of_electricity.csv')
        self.delimiter = ','
    
    def extractData(self):
        tariffData = dict()
        
        with open(self.file, 'r') as csvFile:
            csvReader = csv.reader(csvFile, delimiter=self.delimiter)
            rowIndex = 0
            columnNames = []
            for row in csvReader:
                if 0 == rowIndex:
                    columnNames = row[1:]
                    columnNames = list(map(lambda x : re.split(r'-', x, re.DOTALL), columnNames))
                else:
                    stateAbbreviation = row[0]
                    tariffValues = row[1:]
                    for index in range(len(tariffValues)):
                        if stateAbbreviation not in tariffData:
                            tariffData[stateAbbreviation] = dict()
                        month, year = columnNames[index]
                        if year not in tariffData[stateAbbreviation]:
                            tariffData[stateAbbreviation][year] = dict()
                        tariffData[stateAbbreviation][year][month] = tariffValues[index]
                rowIndex += 1
        
        return tariffData