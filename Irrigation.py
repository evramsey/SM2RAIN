import numpy as np
import pandas as pd

def irrigation_calc(datavals, paramsList):
    t, sm, smdelta, p = datavals
    z, ks, lam, kc = paramsList
    ETpot = kc * (-2 + 1.26*((0.46*t) + 8.13))
    irr = (z * smdelta) + (ks * sm ** (3 + (2/lam))) + (sm * ETpot) - p
    return irr

"""Iterate through all dates and calculates irrigation using parameter values for a given cell; return irrigation values in
a 1-D array"""
def get_cell_irrigation(data, params, start_date, end_date, cell_num):
    irrigation_array = []
    for date, sub_df in data.groupby(level=0):
        # skip first date because of soil moisture delta NaN values
        if date == start_date:
            irrigation_val = 0
        elif date > end_date:
            return irrigation_array
        else:
            precipval = sub_df.loc[date, "precip"][cell_num]
            # calculate irrigation value where precipitation is negligible
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
    return irrigation_array

def get_monthly_irrig_df(data, params, start, end):
    cell_num = 0
    irrig_array = []
    date_list = pd.date_range(start, end)
    while cell_num < data.shape[1]:
        ir = get_cell_irrigation(data, params, start, end, cell_num)
        irrig_array.append(ir)
        cell_num += 1
    irrig_df = pd.DataFrame(irrig_array, columns=date_list, index=data.columns.values)
    irrig_df = irrig_df.transpose()
    irrig_df = irrig_df.groupby(pd.Grouper(freq='M')).sum()
    return irrig_df

def get_daily_irrig_df(data, params, start, end):
    cell_num = 0
    irrig_array = []
    date_list = pd.date_range(start, end)
    while cell_num < data.shape[1]:
        ir = get_cell_irrigation(data, params, start, end, cell_num)
        irrig_array.append(ir)
        cell_num += 1
    irrig_df = pd.DataFrame(irrig_array, columns=date_list, index=data.columns.values)
    irrig_df = irrig_df.transpose()
    # irrig_df = irrig_df.groupby(pd.Grouper(freq='M')).sum()
    return irrig_df