import os
from xlrd import open_workbook

class EnergyConsumptionDataExtractor:
    def __init__(self):
        self.file = os.path.join(os.path.dirname(os.getcwd()), 'EnergyData', 'EnergyConsumptionByStateByMonth.xls')
    
    def extractData(self):
        
        workbook = open_workbook(self.file)
        # Column 0: Year
        # Column 1: Month
        # Column 2: State
        # Column 3: Type of producer
        # Column 4: Energy source
        # Column 5: Consumption
        
        energyConsumptionData = dict()
        
        for sheet in workbook.sheets():
            for rowIndex in range(1, sheet.nrows):
                year = str(int(sheet.cell(rowIndex, 0).value))
                month = str(int(sheet.cell(rowIndex, 1).value))
                state = sheet.cell(rowIndex, 2).value
                typeOfProducer = sheet.cell(rowIndex, 3).value
                energySource = sheet.cell(rowIndex, 4).value
                consumption = sheet.cell(rowIndex, 5).value
                if "2010" == year:
                    continue
                if ("US-TOTAL".lower() == state.lower()) or ("HI".lower() == state.lower()) or ("DC".lower() == state.lower()):
                    continue
                if "total electric power" not in typeOfProducer.lower():
                    continue
                if state not in energyConsumptionData:
                    energyConsumptionData[state] = dict()
                if year not in energyConsumptionData[state]:
                    energyConsumptionData[state][year] = dict()
                if month not in energyConsumptionData[state][year]:
                    energyConsumptionData[state][year][month] = dict()
                if "coal" in energySource.lower():
                    energyConsumptionData[state][year][month]['coal'] = consumption
                elif "petroleum" in energySource.lower():
                    energyConsumptionData[state][year][month]['petroleum'] = consumption
                elif "natural gas" in energySource.lower():
                    energyConsumptionData[state][year][month]['natural gas'] = consumption
        
        return energyConsumptionData