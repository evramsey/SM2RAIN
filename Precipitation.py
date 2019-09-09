import pandas as pd
import numpy as np

'''Calculate the modeled precipitation value based on parameter inputs and temp, sm, and sm delta values
    at a given cell for either a series of dates or a single date'''
def calc_precip(vals, params):
    t, sm, sm_delta = vals
    z, ks, l, kc = params
    evap_pot = kc * (-2 + 1.26 * ((0.46 * t) + 8.13))
    val = (z * sm_delta) + ks * (sm ** (3 + (2 / l))) + (evap_pot * sm)
    if val < 0:
        return 0
    if type(val) == pd.Series:
        val = val.dropna()
        val = [float(x) for x in val]
    return val

'''Iterate through all dates, call calc_precip_cell to calculate precipitation at cell (given by cell_num) 
for all dates; return observed and modeled precipitation, used to generate individual cell model output graph and 
spatial error plot'''
def get_daily_cell_precip_mod_obs(data, start, end, params, cell_num):
    p_calc_array, p_obs_array = [], []
    for date, sub_df in data.groupby(level=0):
        if date >= start and date <= end:
            # get precipitation, temperature, and soil moisture differential values at given date for all cells
            print(date, "cell num:", cell_num)
            temp_val = sub_df.loc[date, "temp"][cell_num]
            sm_val = sub_df.loc[date, "sm"][cell_num]
            sm_delta_val = sub_df.loc[date, "sm_delta"][cell_num]
            precip_val = sub_df.loc[date, "precip"][cell_num]
            val_array = [temp_val, sm_val, sm_delta_val]
            # pass cells to algorithm equation, get back calculated precipitation
            calc_p = calc_precip(val_array, params)
            # keep all precipitation values
            p_calc_array.append(calc_p)
            p_obs_array.append(precip_val)
    return np.array(p_calc_array), np.array(p_obs_array)

'''Calculate precipitation at each cell at each day and return modeled precipitation and observed precipitation'''
def get_opt_precip_arrays(data, params, start_date, end_date):
    i = 0
    calc_precip_array, obs_precip_array = [], []
    while i < data.shape[1]:
        calc_precip, obs_precip = get_daily_cell_precip_mod_obs(data, start_date, end_date,params, i)
        calc_precip_array.append(calc_precip)
        obs_precip_array.append(obs_precip)
        i += 1
    return calc_precip_array, obs_precip_array