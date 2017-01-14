import os
import re

class ClimateDataExtractor:
    def __init__(self):
        self.stateCodeToStateNameMap = {
                    '001': "Alabama",
                    '002': "Arizona",
                    '003': "Arkansas",
                    '004': "California",
                    '005': "Colorado",
                    '006': "Connecticut",
                    '007': "Delaware",
                    '008': "Florida",
                    '009': "Georgia",
                    '010': "Idaho",
                    '011': "Illinois",
                    '012': "Indiana",
                    '013': "Iowa",
                    '014': "Kansas",
                    '015': "Kentucky",
                    '016': "Louisiana",
                    '017': "Maine",
                    '018': "Maryland",
                    '019': "Massachusetts",
                    '020': "Michigan",
                    '021': "Minnesota",
                    '022': "Mississippi",
                    '023': "Missouri",
                    '024': "Montana",
                    '025': "Nebraska",
                    '026': "Nevada",
                    '027': "New Hampshire",
                    '028': "New Jersey",
                    '029': "New Mexico",
                    '030': "New York",
                    '031': "North Carolina",
                    '032': "North Dakota",
                    '033': "Ohio",
                    '034': "Oklahoma",
                    '035': "Oregon",
                    '036': "Pennsylvania",
                    '037': "Rhode Island",
                    '038': "South Carolina",
                    '039': "South Dakota",
                    '040': "Tennessee",
                    '041': "Texas",
                    '042': "Utah",
                    '043': "Vermont",
                    '044': "Virginia",
                    '045': "Washington",
                    '046': "West Virginia",
                    '047': "Wisconsin",
                    '048': "Wyoming",
                    '050': "Alaska"}
        
        self.stateCodeToStateAbbreviationMap = {
                    '001': "AL",
                    '002': "AZ",
                    '003': "AR",
                    '004': "CA",
                    '005': "CO",
                    '006': "CT",
                    '007': "DE",
                    '008': "FL",
                    '009': "GA",
                    '010': "ID",
                    '011': "IL",
                    '012': "IN",
                    '013': "IA",
                    '014': "KS",
                    '015': "KY",
                    '016': "LA",
                    '017': "ME",
                    '018': "MD",
                    '019': "MA",
                    '020': "MI",
                    '021': "MN",
                    '022': "MS",
                    '023': "MO",
                    '024': "MT",
                    '025': "NE",
                    '026': "NV",
                    '027': "NH",
                    '028': "NJ",
                    '029': "NM",
                    '030': "NY",
                    '031': "NC",
                    '032': "ND",
                    '033': "OH",
                    '034': "OK",
                    '035': "OR",
                    '036': "PA",
                    '037': "RI",
                    '038': "SC",
                    '039': "SD",
                    '040': "TN",
                    '041': "TX",
                    '042': "UT",
                    '043': "VT",
                    '044': "VA",
                    '045': "WA",
                    '046': "WV",
                    '047': "WI",
                    '048': "WY",
                    '050': "AK"}
        
        self.fileMapping = {
                        # 'coolingDegreeDays': "climdiv-cddcst-v1.0.0-20161104",
                        # 'heatingDegreeDays': "climdiv-hddcst-v1.0.0-20161104",
                        'precipitation': "climdiv-pcpnst-v1.0.0-20161104",
                        'maximumTemperature': "climdiv-tmaxst-v1.0.0-20161104",
                        'minimumTemperature': "climdiv-tminst-v1.0.0-20161104",
                        'averageTemperature': "climdiv-tmpcst-v1.0.0-20161104"}
        
        self.yearLowerBound = 2001
        self.yearHigherBound = 2010
    
    def extractData(self):
        climateData = dict()
        # {stateCode_i: {year_j: {month_k: [feature_x: value_month_x, ...]}, ...}, ...}
        
        for feature in self.fileMapping:
            file = os.path.join(os.path.dirname(os.getcwd()), 'ClimateData', self.fileMapping[feature])
            with open(file, 'r') as inputFeatureFile:
                for line in inputFeatureFile:
                    stateCode = line[0:3]
                    division = line[3]
                    elementCode = line[4:6]
                    year = int(line[6:10])
                    monthWiseReadings = re.split(r'\s+', line[11:].strip(), re.DOTALL)
                    monthWiseReadings = list(map(lambda x : float(x), monthWiseReadings))
                    if stateCode not in self.stateCodeToStateAbbreviationMap:
                        continue
                    stateAbbreviation = self.stateCodeToStateAbbreviationMap[stateCode]
                    if (year >= self.yearLowerBound) and (year <= self.yearHigherBound):
                        if stateAbbreviation not in climateData:
                            climateData[stateAbbreviation] = dict()
                        if str(year) not in climateData[stateAbbreviation]:
                            climateData[stateAbbreviation][str(year)] = dict()
                        for monthIndex in range(1, len(monthWiseReadings) + 1):
                            if str(monthIndex) not in climateData[stateAbbreviation][str(year)]:
                                climateData[stateAbbreviation][str(year)][str(monthIndex)] = dict()
                            climateData[stateAbbreviation][str(year)][str(monthIndex)][feature] = monthWiseReadings[monthIndex - 1]
        
        return climateData