import Input_Reader as rdr
from sklearn.metrics import mean_squared_error
import numpy as np
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os, datetime

# Testing files
# temp = rdr.getAvgTemp("air_temp_avg_2003_08_GLDAS.csv")
# precip = rdr.getPrecip("totalprecipitation_2003_08_GLDAS.csv")
# sm = rdr.getTopSoilMoisture("top_soil_moisture_agg_2003_08_GLDAS.csv")
# datelist = pd.date_range(pd.datetime(2003, 8, 1), pd.datetime(2003, 8, 31))

today = datetime.datetime.today().strftime('%Y-%m-%d')

#full data set
# dailydatelist = pd.date_range(pd.datetime(2003, 1, 1), pd.datetime(2011, 12, 31))
# monthlydatelist = pd.date_range(pd.datetime(2003, 1, 1), pd.datetime(2011, 12, 31), freq='MS')
# yearlist = [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011]
# temp = rdr.getAvgTemp("Air_Temp_Composite_GLDAS.csv") - 273.15   # GLDAS temperature in Kelvin, converting to Celsius
# precip = rdr.getPrecip("Total_Precip_Composite_GLDAS.csv")   # GLDAS precip in kg/day, which is equivalent to mm/day
# sm = rdr.getTopSoilMoisture("Top_Soil_Moisture_Composite_GLDAS.csv")   #GLDAS soil moisture in m^3/m^3

#training data set
dailydatelist = pd.date_range(pd.datetime(2003, 1, 1), pd.datetime(2009, 12, 31))
monthlydatelist = pd.date_range(pd.datetime(2003, 1, 1), pd.datetime(2009, 12, 31), freq='MS')
yearlist = [2003, 2004, 2005, 2006, 2007, 2008, 2009]
temp = rdr.getAvgTemp("Air_Temp_Composite_GLDAS_training.csv") - 273.15   # GLDAS temperature in Kelvin, converting to Celsius
precip = rdr.getPrecip("Total_Precip_Composite_GLDAS_training.csv")   # GLDAS precip in kg/day, which is equivalent to mm/day
sm = rdr.getTopSoilMoisture("Top_Soil_Moisture_Composite_GLDAS_training.csv")   #GLDAS soil moisture in m^3/m^3

data = pd.concat([temp, precip, sm, sm.diff()], keys=['temp', 'precip', 'sm', 'sm_delta'], names=['source'])
data.reset_index(inplace=True)
data.set_index(['Date', 'source'], inplace=True)
data.dropna()
data.sort_index(inplace=True)
numCells = data.shape[1]
print(data)

rmsList, predictedvalsList, actualvalsList, paramsList = [], [], [], []

def calc_precip(vals, params):
    """Calculates the modeled precipitation value based on parameter inputs and temp, sm, and sm delta values for all
    cells at a given date"""
    t, sm, sm_delta = vals
    z, ks, l, kc = params
    ETpot = kc * (-2 + 1.26*((0.46*t) + 8.13))
    val = (z*sm_delta) + ks*(sm**(3+(2/l))) + (ETpot*sm)
    val = val.dropna()
    val_f = [float(x) for x in val]
    return val_f

def getJustObsData(cell_num):
    vals = []
    for date, sub_df in data.groupby(level=0):
        precipvals = sub_df.loc[date, "precip"][cell_num]
        vals.append(precipvals)
    return vals

def obj_func(params):
    """Iterates through all dates, pulls all data points for cells in which observed precip > 0, and
    calculates precip using parameter values; returns model precip and observed precip arrays"""
    p_calc_array, p_obs_array = [], []
    print("objective function calculating for: ")
    for date, sub_df in data.groupby(level=0):
        calc_p = []
        # if precip > 0, get precip, temp, sm and sm delta values at given date for all cells
        if any(sub_df.loc[date, "precip"] > 0.001):
            tempvals = sub_df.loc[date, "temp"]
            smvals = sub_df.loc[date, "sm"]
            smdeltavals = sub_df.loc[date, "sm_delta"]
            precipvals = sub_df.loc[date, "precip"]
            val_array = [tempvals, smvals, smdeltavals]
            # pass cells to model equation, get back calculated precipitation
            calc_p = calc_precip(val_array, params)
        # add all values where precip = 0 to the rest of the array
        if calc_p != []:
            p_calc_array.append(calc_p)
            p_obs_array.append([float(x) for x in precipvals])
    p_calc_array = np.array(p_calc_array).flatten()
    p_obs_array = np.array(p_obs_array).flatten()
    return p_calc_array, p_obs_array

def get_rms(params):
    """Calculates the root mean squared error between modeled values and observed data"""
    p_predicted, p_actual = obj_func(params)
    rms = np.sqrt(mean_squared_error(p_actual, p_predicted))
    print("rms: ", rms)
    rmsList.append(rms)
    predictedvalsList = p_predicted
    actualvalsList = p_actual
    paramsList.append(params)
    return rms

# def minimizeBH(init_guess, step_size):
#     """sets bounds on parameters and minimizes the output of the root mean square function by
#     varying parameters"""
#     bnds =((0,1000), (0, 12200), (0.00001, 5), (None, None))
#     # bounds = ((0,1), (0, 12200), (0.000001, 5), (-1, 1))
#     # results = opt.minimize(get_rms, init_guess, bounds=bnds, tol = 1e-5)
#     args = {"bounds": bnds, "tol": 1e-5}
#     results = opt.basinhopping(get_rms, init_guess, niter=100, minimizer_kwargs=args, stepsize=step_size)
#     print(results)
#     saveVals(str(step_size))
#     return results.x

def minimizeNM(init_guess):
    """sets bounds on parameters and minimizes the output of the root mean square function by
    varying parameters"""
    bnds =((0,1000), (0, 12200), (0.00001, 5), (None, None))
    # results = opt.minimize(get_rms, init_guess, bounds=bnds, tol = 1e-5)
    args = {"bounds": bnds, "tol": 1e-5}
    results = opt.minimize(get_rms, init_guess, method='Nelder-Mead')
    print(results)
    saveVals("Local_Optimizer_" + today)
    return results.x

def minimizeGA():
    """sets bounds on parameters and minimizes the output of the root mean square function by
    varying parameters"""
    bnds =((0, 1000), (0, 12200), (0.00001, 5), (-1, 1))
    # results = opt.minimize(get_rms, init_guess, bounds=bnds, tol = 1e-5)
    args = {"bounds": bnds, "tol": 1e-5}
    results = opt.differential_evolution(get_rms, bnds)
    print(results)
    saveVals("GA_Optimizer_" + today)
    return results.x

def minimizePowell(init_guess, name):
    """sets bounds on parameters and minimizes the output of the root mean square function by
    varying parameters"""
    bnds =((0,1000), (0, 12200), (0.00001, 5), (None, None))
    # results = opt.minimize(get_rms, init_guess, bounds=bnds, tol = 1e-5)
    args = {"bounds": bnds, "tol": 1e-5}
    results = opt.minimize(get_rms, init_guess, method='Powell')
    print(results)
    saveVals("Powell_Optimizer_" + today + "_" + name)
    return results.x

def calc_precip_cell(vals, params):
    """Calculates precipitation at a single cell at a single date"""
    t, sm, sm_delta = vals
    z, ks, l, kc = params
    ETpot = kc * (-2 + 1.26*((0.46*t) + 8.13))
    val = (z*sm_delta + ks*sm**(3+(2/l)) + ETpot*sm)
    return val

def getDailyCellPrecip(params, cell_num):
    """Iterates through all dates, calls calc_precip_cell to calculate precipitation at cell (given by cell_num)
    for all dates; used to generate model output graph"""
    p_calc_array, p_obs_array = [], []
    print("objective function calculating for: ")
    for date, sub_df in data.groupby(level=0):
        # get precipitation, temperature, and soil moisture differential values at given date for all cells
        print(date)
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
    return p_calc_array, p_obs_array

def plotDailyCellPrecip(params, cell_num):
    p_calc, p_obs = getDailyCellPrecip(params, cell_num)
    plt.plot(dailydatelist, p_calc, "b", dailydatelist, p_obs, "g")
    plt.legend([p_calc, p_obs], ["Modeled Precipitation", "Actual Precipitation"])
    plt.xlabel("Time")
    plt.ylabel("Precipitation (mm/day)")
    plt.show()

def plotMonthlyCellPrecip(params, cell_num):
    p_calc, p_obs = getDailyCellPrecip(params, cell_num)
    calc_df = pd.DataFrame(p_calc, index=dailydatelist)
    calc_df = calc_df.groupby(pd.TimeGrouper(freq='M')).sum()
    obs_df = pd.DataFrame(p_obs, index=dailydatelist)
    obs_df = obs_df.groupby(pd.TimeGrouper(freq='M')).sum()
    calc_df.loc[calc_df[0] < 0, 0] = 0
    plt.plot(monthlydatelist, calc_df, "#4DCAF4", monthlydatelist, obs_df, "#FFCE9A")
    model_line = mlines.Line2D([], [], color='#4DCAF4', label="Modeled Precipitation")
    obs_line = mlines.Line2D([],[], color='#FFCE9A', label="Observed Precipitation")
    plt.legend(handles=[model_line, obs_line])
    plt.xlabel("Time")
    plt.ylabel("Precipitation (mm/month)")
    plt.show()

def plotAverageObservedBasinPrecip(temp_res):
    '''plot line graph of average observed precipitation for all cells for either
    monthly or annual resolution'''
    # sum daily precipitation for each cell to get monthly or annual precipitation, then take the average
    # across all cells
    if temp_res == "month":
        frequency = 'M'
    elif temp_res == "year":
        frequency = 'Y'
    else:
        print("Check temporal resolution in Average Observed Basin Precipitation function")
    avgPrecip = precip.groupby(pd.Grouper(freq=frequency)).sum().mean(axis=1)
    avgPrecip = avgPrecip.tolist()
    overallAverage = np.mean(avgPrecip)
    print(overallAverage)
    print(avgPrecip)
    #
    # if temp_res == "month":
    #     plt.plot(monthlydatelist, avgPrecip, color="#FBB652")
    # else:
    #     plt.plot(yearlist, avgPrecip, color="#FBB652")
    # plt.axhline(overallAverage)
    # plt.xlabel("Time")
    # plt.ylabel("Precipitation (mm/{0})".format(temp_res))
    # plt.show()
    return avgPrecip

def saveVals(folderName):
    os.mkdir(folderName)
    rmsdf = pd.DataFrame(rmsList)
    predvalsdf = pd.DataFrame(predictedvalsList)
    paramsdf = pd.DataFrame(paramsList)
    rmsdf.to_csv(folderName + "/rms_vals_" + today +".csv", header=None)
    predvalsdf.to_csv(folderName + "/predicted_vals_" + today + ".csv", header=None)
    paramsdf.to_csv(folderName +"/parameter_vals_" + today + ".csv", header=None)

def saveSolutions(res, size, tottime):
    solutions = '  '.join([str(x) for x in res.x])
    s = "Solutions (Z*, Ks, Lambda, Kc): " + solutions
    message = '  '.join([str(x) for x in res.message])
    s+= "\nMessage: " + message
    s+= "\nFunction objective value: " + str(res.fun)
    s+= "\nNumber of evaluations: " + str(res.nfev)
    s+= "\nNumber of iterations: " + str(res.nit)
    s+= "\nBasin size: " + str(size)
    s+= "\nTime to run (in seconds): " + str(tottime)
    txtFileName = "basin_" + str(size) + "opt_results_" + today + ".txt"
    with open(txtFileName, "w") as text_file:
        text_file.write(s)
    print(s)



# these puppies are the best performers for wrong cell set
nm_basin_params = [2.32461119e+02, 1.46352926e+02, 3.16673194e+00, 7.22414553e-02]
nm_ga_params = [2.32461114e+02, 1.46352973e+02, 3.16673071e+00, 7.22414408e-02]


# minimizeGA()
ga_params = [2.33274822e+02, 8.14493074e+02, 8.58872370e-01, 8.67620061e-02]
minimizeNM(ga_params)

np_params = [2.33312910e+02, 6.27836656e+02, 9.74982222e-01, 7.33952104e-02]
#plotMonthlyCellPrecip(params_training, 3)
#plotAverageObservedBasinPrecip("month")