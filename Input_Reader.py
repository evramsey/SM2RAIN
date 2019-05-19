import os
import pandas as pd

rootDir = os.path.dirname(__file__)
upDir = os.path.dirname(rootDir)
inputDir = upDir + "/inputs/"
error_msg = "Error in {0} input file: data source not recognized"

mydateparser = lambda x: pd.datetime.strptime(x, "%Y-%m-%d")

def getAvgTemp(tempFile):
    inFile = inputDir + tempFile
    if "CPC" in inFile:
        tempdf = pd.read_csv(inFile, index_col=[0, 1], parse_dates=True, date_parser=mydateparser)
        tempdf = tempdf.drop(columns=["Min Temp", "Max Temp"])
    elif "GLDAS" in inFile:
        tempdf = pd.read_csv(inFile, index_col=0, parse_dates=True, date_parser=mydateparser)
    else:
        print(error_msg.format("temperature"))
        SystemExit
    return tempdf

def getPrecip(precipFile):
    inFile = inputDir + precipFile
    if "TRMM" in inFile:
        precipdf = pd.read_csv(inFile, index_col=[0,1], parse_dates=True, date_parser=mydateparser)
        precipdf = precipdf.xs("precipitation", level=1, drop_level=False, date_parser=mydateparser)
    elif "GLDAS" in inFile:
        precipdf = pd.read_csv(inFile, index_col=0, parse_dates=True, date_parser=mydateparser)
    else:
        print(error_msg.format("precipitation"))
        SystemExit
    return precipdf

def getTopSoilMoisture(smFile):
    inFile = inputDir + smFile
    if "GLDAS" in inFile:
        smdf = pd.read_csv(inFile, index_col=0, parse_dates=True, date_parser=mydateparser)
        # smdf - smdf["Date"].to_datetime(format="%m%d%Y")
        # convert kg/m^2 to volumetric soil moisture (m^3/m^3) by dividing by layer thickness (100 mm)
        smdf = smdf/100
        return smdf
    else:
        print(error_msg.format("soil moisture"))
        SystemExit

def getIrrigation(irrigFile):
    inFile = inputDir + irrigFile
    irrig = pd.read_csv(inFile, index_col=0, parse_dates=True, date_parser=mydateparser)
    return irrig

def getErrorVals(errorFile):
    inFile = inputDir + errorFile
    errorFile = pd.read_csv(inFile, index_col=0)
    return errorFile