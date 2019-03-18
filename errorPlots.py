import numpy as np
import Input_Reader as rdr
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import seaborn as sns; sns.set()

# Testing files
# temp = rdr.getAvgTemp("air_temp_avg_2003_08_GLDAS.csv")
# precip = rdr.getPrecip("totalprecipitation_2003_08_GLDAS.csv")
# sm = rdr.getTopSoilMoisture("top_soil_moisture_agg_2003_08_GLDAS.csv")
# dailydatelist = pd.date_range(pd.datetime(2003, 8, 1), pd.datetime(2003, 8, 31))
# errordailydatelist = pd.date_range(pd.datetime(2003, 8, 2), pd.datetime(2003, 8, 31))


params = [2.32624657e+02, 3.04321402e+02, 1.38520082e+00, 1.33248558e-01]
params_training = [2.33336161e+02, 1.25603181e+02, 3.94587645e+00, 6.69247287e-02]

dailydatelist = pd.date_range(pd.datetime(2003, 1, 1), pd.datetime(2011, 12, 31))
errordailydatelist = pd.date_range(pd.datetime(2003, 1, 2), pd.datetime(2011, 12, 31))
monthlydatelist = pd.date_range(pd.datetime(2003, 1, 1), pd.datetime(2011, 12, 31), freq='MS')
monthlydatelist = monthlydatelist.strftime("%m-%Y")
# temp = rdr.getAvgTemp("Air_Temp_Composite_GLDAS.csv") - 273.15   # GLDAS temperature in Kelvin, converting to Celsius
# precip = rdr.getPrecip("Total_Precip_Composite_GLDAS.csv")   # GLDAS precip in kg/day, which is equivalent to mm/day
# sm = rdr.getTopSoilMoisture("Top_Soil_Moisture_Composite_GLDAS.csv")   #GLDAS soil moisture in m^3/m^3
temp = rdr.getAvgTemp("Air_Temp_Composite_GLDAS_prediction.csv") - 273.15   # GLDAS temperature in Kelvin, converting to Celsius
precip = rdr.getPrecip("Total_Precip_Composite_GLDAS_prediction.csv")   # GLDAS precip in kg/day, which is equivalent to mm/day
sm = rdr.getTopSoilMoisture("Top_Soil_Moisture_Composite_GLDAS_prediction.csv")   #GLDAS soil moisture in m^3/m^3
data = pd.concat([temp, precip, sm, sm.diff()], keys=['temp', 'precip', 'sm', 'sm_delta'], names=['source'])
data.reset_index(inplace=True)
data.set_index(['Date', 'source'], inplace=True)
data.dropna()
data.sort_index(inplace=True)
numCells = data.shape[1]
print(data)

def calc_precip_cell(vals, params):
    """Calculates precipitation at a single cell at a single date"""
    t, sm, sm_delta = vals
    z, ks, l, kc = params
    ETpot = kc * (-2 + 1.26*((0.46*t) + 8.13))
    val = (z*sm_delta + ks*sm**(3+(2/l)) + ETpot*sm)
    return val

def getDailyPrecipAtCell(params, cell_num):
    """Iterates through all dates, calls calc_precip_cell to calculate precipitation at cell (given by cell_num)
    for all dates; used to generate model output graph and spatial error plot"""
    p_calc_array, p_obs_array = [], []
    for date, sub_df in data.groupby(level=0):
        # get precipitation, temperature, and soil moisture differential values at each date for all cells
        tempval = sub_df.loc[date, "temp"][cell_num]
        smval = sub_df.loc[date, "sm"][cell_num]
        smdeltaval = sub_df.loc[date, "sm_delta"][cell_num]
        precipval = sub_df.loc[date, "precip"][cell_num]
        val_array = [tempval, smval, smdeltaval]
        # pass cells to algorithm equation, get back calculated precipitation
        calc_p = calc_precip_cell(val_array, params)
        # keep all precipitation values
        p_calc_array.append(calc_p)
        p_obs_array.append(precipval)
    p_calc_array = np.array(p_calc_array)
    p_obs_array = np.array(p_obs_array)
    return p_calc_array, p_obs_array

def getTemporalErrorArray():
    errorArray = []
    num = 0
    while num < numCells:
        precip_model, precip_obs = getDailyPrecipAtCell(params, num)
        # remove first value of array, which is NaN because of soil moisture delta
        precip_model = np.delete(precip_model, 0)
        precip_obs = np.delete(precip_obs, 0)
        error = np.abs(precip_obs - precip_model)
        errorArray.append(error)
        num += 1
        print(num)
    # print(np.shape(errorArray))
    # print(errorArray)
    errorDF = pd.DataFrame(errorArray, columns=errordailydatelist, index=range(0, numCells))
    error_df = errorDF.transpose()
    grouped_error_df = error_df.groupby(pd.Grouper(freq='M'))
    errorList = []
    for name, val in grouped_error_df:
        valArray = np.array(val.values.tolist())
        valArray = valArray.flatten()
        errorList.append(valArray)
    return errorList

def createBoxPlot(data_to_plot, outFileName):
    boxColors = ['#01ACD2', '#FB822A']    # blue-green and orange
    outlineColor = 'black'
    fig, ax1 = plt.subplots()
    fig.set_size_inches(6, 3)
    bp = plt.boxplot(data_to_plot, widths = 0.8, showfliers=False, patch_artist=True)
    # for box in bp['boxes']:
    #     count = 0
    #     box.set(color=outlineColor, linewidth=1)
    #     if count < 84:
    #         box.set(facecolor=boxColors[0])
    #     else:
    #         box.set(facecolor=boxColors[1])
    #     count += 1
    for box in bp['boxes']:
        count = 0
        if count < 84:
            k = 0
        else:
            k = 1
        box.set(facecolor=boxColors[k])
        count += 1
    for median in bp['medians']:
        median.set(color=outlineColor, linewidth=1)
    for whisker in bp['whiskers']:
        whisker.set(color=outlineColor, linewidth=1)
    # for flier in bp['fliers']:
    #     flier.set(marker='x', color='#FFCE9A', alpha=0.5)
    ax1.set_ylabel('Error (mm/day)', fontsize=18)
    ax1.set_facecolor('#FFFFFF')
    ax1.spines['bottom'].set_color('black')
    ax1.spines['left'].set_color('black')
    ax1.tick_params(axis='both', which='both', direction='out', colors='black', length=3, width=1)
    for tick in ax1.get_yticklabels():
        tick.set_fontsize(fontsize=14)
    tickrange = range(0, 108, 6)
    ticklabels = ["2003-Jan", "2003-Jul", "2004-Jan", "2004-Jul",
                  "2005-Jan", "2005-Jul", "2006-Jan", "2006-Jul",
                  "2007-Jan", "2007-Jul", "2008-Jan", "2008-Jul",
                  "2009-Jan", "2009-Jul", "2010-Jan", "2010-Jul",
                  "2011-Jan", "2011-Jul"]
    #xtickNames = plt.setp(ax1, xticklabels=monthlydatelist
    plt.xticks(tickrange, ticklabels, fontsize=14, rotation=30)
    ax1.set_xlabel('Time', fontsize=18)
    fig.savefig(outFileName, bbox_inches='tight')
    #fig.savefig(outFileName)
    plt.show()

def createTemporalErrorArrayCSV(fileName):
    errorList = getTemporalErrorArray()
    errorDF = pd.DataFrame(errorList)
    errorDF.to_csv(fileName)

def getBoxPlotData(errorDF):
    n = 0
    plot_data = []
    while n < len(monthlydatelist):
        testline = errorDF.iloc[n]
        testline = testline.dropna()
        monthlyerrorvals = testline.values.tolist()
        plot_data.append(monthlyerrorvals)
        n += 1
    return plot_data

def getHeatMapRMS(cellListLine):
    heatMapLine = []
    for cell in cellListLine:
        p_predicted, p_actual = getDailyPrecipAtCell(params, cell)
        p_predicted = np.delete(p_predicted, 0)
        p_actual = np.delete(p_actual, 0)
        rms = np.sqrt(mean_squared_error(p_actual, p_predicted))
        heatMapLine.append(rms)
    return heatMapLine

def getPrecipErrorAtAllCells(params):
    correctCellListLine0 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    line0_RMS = getHeatMapRMS(correctCellListLine0)
    correctCellListLine1 = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
    line1_RMS = getHeatMapRMS(correctCellListLine1)
    correctCellListLine2 = [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43]
    line2_RMS = getHeatMapRMS(correctCellListLine2)
    correctCellListLine3 = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
    line3_RMS = getHeatMapRMS(correctCellListLine3)
    correctCellListLine4 = [64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75]
    line4_RMS = getHeatMapRMS(correctCellListLine4)
    matrx = [line0_RMS, line1_RMS, line2_RMS, line3_RMS, line4_RMS]
    colors = ['#C9F0FC', '#A1E0F3', '#79D0EB', '#51C0E2', '#29B0DA', '#01A0D2']
    palette = sns.color_palette(colors)
    ax = sns.heatmap(matrx, square=True, cmap=palette, cbar=True, cbar_kws=dict(use_gridspec=False, location="bottom"))
    plt.show()
    #identify the cells that are *actually* on the map by number
    #create an empty list

#createTemporalErrorArrayCSV("ErrorArrayForBoxPlot25Feb.csv")
errorDF = rdr.getErrorVals("ErrorArrayForBoxPlot25Feb.csv")
createBoxPlot(getBoxPlotData(errorDF), "Feb27BoxPlot.png")

# getPrecipErrorAtAllCells(params_training)

