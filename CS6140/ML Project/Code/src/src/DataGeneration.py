from ClimateDataExtraction import ClimateDataExtractor
from PopulationDataExtraction import PopulationDataExtractor
from EnergyTariffDataExtraction import EnergyTariffDataExtractor
from GdpDataExtraction import GdpDataExtractor
from EnergyConsumptionDataExtraction import EnergyConsumptionDataExtractor

class DataGenerator:
    def __init__(self):
        self.climateData = ClimateDataExtractor().extractData()
        print("Loaded climateData file")
        self.populationData = PopulationDataExtractor().extractData()
        print("Loaded populationData file")
        self.tariffData = EnergyTariffDataExtractor().extractData()
        print("Loaded tariffData file")
        self.gdpData = GdpDataExtractor().extractData()
        print("Loaded gdpData file")
        self.energyConsumptionData = EnergyConsumptionDataExtractor().extractData()
        print("Loaded energyConsumptionData file")
    
    def generateData(self):
        data = dict()
        labels = dict()
        for state in list(sorted(self.energyConsumptionData.keys())):
            data[state] = dict()
            labels[state] = dict()
            for year in list(sorted(self.energyConsumptionData[state].keys())):
                data[state][year] = dict()
                labels[state][year] = dict()
                months = list(sorted(map(lambda x : int(x), self.energyConsumptionData[state][year])))
                for month in months:
                    row = []
                    monthString = str(month)
                    row.append(self.climateData[state][year][monthString]['averageTemperature'])
                    row.append(self.climateData[state][year][monthString]['minimumTemperature'])
                    row.append(self.climateData[state][year][monthString]['maximumTemperature'])
                    row.append(self.climateData[state][year][monthString]['precipitation'])
                    row.append(self.populationData[state][year])
                    row.append(self.tariffData[state][year][monthString])
                    row.append(self.gdpData[state][year])
                    data[state][year][monthString] = row
                    row = []
                    # print(state + ", " + year + ", " + monthString + ", " + str(self.energyConsumptionData[state][year][monthString]))
                    if 'coal' in self.energyConsumptionData[state][year][monthString]:
                        row.append(self.energyConsumptionData[state][year][monthString]['coal'])
                    else:
                        row.append(0.0)
                    if 'petroleum' in self.energyConsumptionData[state][year][monthString]:
                        row.append(self.energyConsumptionData[state][year][monthString]['petroleum'])
                    else:
                        row.append(0.0)
                    if 'natural gas' in self.energyConsumptionData[state][year][monthString]:
                        row.append(self.energyConsumptionData[state][year][monthString]['natural gas'])
                    else:
                        row.append(0.0)
                    labels[state][year][monthString] = row
        return {'data': data, 'labels': labels}

dataGenerator = DataGenerator()
dataGenerator.generateData()