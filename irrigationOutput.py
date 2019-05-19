import Input_Reader as rdr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import algorithm as alg
import matplotlib.lines as mlines

# Testing files
# temp = rdr.getAvgTemp("air_temp_avg_2003_08_GLDAS.csv")
# precip = rdr.getPrecip("totalprecipitation_2003_08_GLDAS.csv")
# sm = rdr.getTopSoilMoisture("top_soil_moisture_agg_2003_08_GLDAS.csv")
# datelist = pd.date_range(pd.datetime(2003, 8, 1), pd.datetime(2003, 8, 31))
# monthlydatelist = pd.date_range(pd.datetime(2003, 8, 1), pd.datetime(2003, 8, 31), freq='MS')

paramsList = [2.33336161e+02, 1.25603181e+02, 3.94587645e+00, 6.69247287e-02]

temp = rdr.getAvgTemp("Air_Temp_Composite_GLDAS.csv") - 273.15   # GLDAS temperature in Kelvin, converting to Celsius
precip = rdr.getPrecip("Total_Precip_Composite_GLDAS.csv")   # GLDAS precip in kg/day, which is equivalent to mm/day
sm = rdr.getTopSoilMoisture("Top_Soil_Moisture_Composite_GLDAS.csv")   #GLDAS soil moisture in m^3/m^3
datelist = pd.date_range(pd.datetime(2003, 1, 1), pd.datetime(2011, 12, 31))
monthlydatelist = pd.date_range(pd.datetime(2003, 1, 1), pd.datetime(2011, 12, 31), freq='MS')

data = pd.concat([temp, precip, sm, sm.diff()], keys=['temp', 'precip', 'sm', 'sm_delta'], names=['source'])
data.reset_index(inplace=True)
data.set_index(['Date', 'source'], inplace=True)
#data.dropna()
data.sort_index(inplace=True)
print(data)
numCells = data.shape[1]


def irrigation_calc(datavals, paramsList):
    t, sm, smdelta, p = datavals
    z, ks, lam, kc = paramsList
    ETpot = kc * (-2 + 1.26*((0.46*t) + 8.13))
    irr = (z * smdelta) + (ks * sm ** (3 + (2/lam))) + (sm * ETpot) - p
    return irr

def irrigation(params, cell_num):
    """Iterates through all dates and calculates irrigation using parameter values for a given cell; returns irrigation values in
    a 1-D array"""
    irrigation_array = []
    for date, sub_df in data.groupby(level=0):
        # skip first date because of soil moisture delta NaN values
        if date == "2003-01-01":
            irrigation_val = 0
        else:
            precipval = sub_df.loc[date, "precip"][cell_num]
            if sub_df.loc[date, "precip"][cell_num] < 0.001:
                tempval = sub_df.loc[date, "temp"][cell_num]
                smval = sub_df.loc[date, "sm"][cell_num]
                smdeltaval = sub_df.loc[date, "sm_delta"][cell_num]
                val_array = [tempval, smval, smdeltaval, precipval]
            # pass cells to model equation, get back calculated irrigation
                irrigation_val = irrigation_calc(val_array, params)
                if irrigation_val < 0:
                    irrigation_val = 0
            else:
                irrigation_val = 0
        irrigation_array.append(irrigation_val)
    irrigation_array = np.array(irrigation_array)
    return irrigation_array

def getMonthlyCellPrecip(cell_num):
    precip_vals = precip.groupby(pd.Grouper(freq='M')).sum()
    precip_val = precip_vals.iloc[:,cell_num]
    return precip_val

def filterIrrigationForRatio(precip, irrig):
    filtered_irrig = []
    count = 0
    irrig = irrig.to_numpy()
    for irrig_val in irrig:
        if precip[count] == 0:
            filtered_irrig.append(np.asscalar(irrig_val))
        elif irrig_val/precip[count] < 1.0:
            filtered_irrig.append(0)
        else:
            filtered_irrig.append(np.asscalar(irrig_val))
        count += 1
    return filtered_irrig

def getFilteredIrrigationForCell(params, cell_num):
    irrig = irrigation(params, cell_num)
    irrig_df = pd.DataFrame(irrig, index=datelist)
    irrig_df = irrig_df.groupby(pd.Grouper(freq="M")).sum()
    precip_df = getMonthlyCellPrecip(cell_num)
    filtered_irrig = filterIrrigationForRatio(precip_df, irrig_df)
    return filtered_irrig

def plotFilteredCellIrrigation(params, cell_num):
    monthlydatelist = pd.date_range(pd.datetime(2003, 1, 1), pd.datetime(2011, 12, 31), freq='MS')
    filtered_irrig = getFilteredIrrigationForCell(params, cell_num)
    plt.rcParams["figure.figsize"] = [50, 5]
    plt.plot(monthlydatelist, filtered_irrig, "#01ACD2")
    plt.xlabel("Time")
    plt.ylabel("Irrigation (mm/month)")
    plt.show()

cellNum = 0
listicle = []
while cellNum < numCells:
    irrig = getFilteredIrrigationForCell(paramsList, cellNum)
    listicle.append(irrig)
    cellNum += 1
    print("cellNum:", cellNum)
print(listicle)
irrig_df = pd.DataFrame(listicle, columns=monthlydatelist)
print("----")
print(irrig_df.mean(axis=0))
mean_irrig = irrig_df.mean(axis=0)
avgPrecip = alg.plotAverageObservedBasinPrecip("month")
plt.plot()
plt.plot(monthlydatelist, mean_irrig, "#4FD093", monthlydatelist, avgPrecip, "#FBB652")
model_line = mlines.Line2D([], [], color='#4FD093', label="Modeled Irrigation")
obs_line = mlines.Line2D([], [], color='#FBB652', label="Observed Precipitation")
plt.legend(handles=[model_line, obs_line])
plt.xlabel("Time")
plt.ylabel("mm/month")
plt.show()