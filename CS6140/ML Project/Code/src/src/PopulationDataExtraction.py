import os
from xlrd import open_workbook

class PopulationDataExtractor:
    def __init__(self):
        self.file = os.path.join(os.path.dirname(os.getcwd()), 'PopulationData', 'ST-EST00INT-01.xls')
    
    def extractData(self):
        workbook = open_workbook(self.file)

        populationData = dict()
        
        for sheet in workbook.sheets():
            for rowIndex in range(1, sheet.nrows):
                for year, columnIndex in zip(sheet.row(0), range(sheet.ncols)):
                    if 0 == columnIndex:
                        stateAbbreviation = sheet.cell(rowIndex, columnIndex).value
                        populationData[stateAbbreviation] = dict()
                        continue
                    populationData[stateAbbreviation][str(int(year.value))] = int(sheet.cell(rowIndex, columnIndex).value)
        
        return populationData