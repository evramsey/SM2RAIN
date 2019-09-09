import InputReader as rdr
import Precipitation as prcp
from sklearn.metrics import mean_squared_error
import numpy as np
import scipy.optimize as opt
import pandas as pd
import os, sys
import datetime
from time import process_time
import ErrorPlotGenerator as epg
import OutputPlotGenerator as opg
import Irrigation as irrig

rms_list, params_list = [], []

def find_optimal_parameters(data, trng_start_date, trng_end_date, vld_start_date, vld_end_date, optimizer1, bas_size,
                            optimizer2, out_flag, irrig_csv, obs_file_name, mod_file_name):
    today = datetime.datetime.today().strftime('%Y-%m-%d %H:%S')
    irrig_fldr = os.path.dirname(irrig_csv)
    '''Iterate through all dates, pull all data points for cells in which observed precip > 0, and
    calculate precipitation using given parameter values; return model and observed precipitation arrays'''
    def obj_func(params):
        p_calc_array, p_obs_array = [], []
        for date, sub_df in data.groupby(level=0):
            if date >= trng_start_date and date <= vld_end_date:
                calc_p = []
                # if precip > 0, get precip, temp, sm and sm delta values at given date for all cells
                if any(sub_df.loc[date, 'precip'] > 0.001):
                    temp_vals = sub_df.loc[date, 'temp']
                    sm_vals = sub_df.loc[date, 'sm']
                    sm_delta_vals = sub_df.loc[date, 'sm_delta']
                    precip_vals = sub_df.loc[date, 'precip']
                    val_array = [temp_vals, sm_vals, sm_delta_vals]
                    # pass cells to model equation, get back calculated precipitation
                    calc_p = prcp.calc_precip(val_array, params)
                # add all values where precip = 0 to the rest of the array
                if calc_p != []:
                    p_calc_array.append(calc_p)
                    p_obs_array.append([float(x) for x in precip_vals])
        print(np.shape(p_calc_array))
        p_calc_array = np.array(p_calc_array).transpose()
        p_obs_array = np.array(p_obs_array).transpose()
        return p_calc_array, p_obs_array

    '''Calculate the root mean squared error between modeled values and observed data, add
        the root mean square error to a list, and revise predicted values list. '''
    def get_trng_rms(params):
        global rms_list
        global params_list
        p_predicted, p_obs = obj_func(params)
        rms = np.sqrt(mean_squared_error(p_obs, p_predicted))
        print('rms: ' + str(rms))
        rms_list.append(rms)
        params_list.append(params)
        return rms

    def run_selected_optimizers(glob_opt, loc_opt):
        t = process_time()
        bnds = ((0, 1000), (0, 12200), (0.00001, 5), (-1, 1))
        tol = 1e-5
        initial_param_guess = [100, 100, 1, 0]
        if glob_opt.lower() == 'genetic algorithm' or glob_opt.lower() == 'ga':
            global_results = opt.differential_evolution(get_trng_rms, bounds=bnds, tol=tol)
        elif ("basin" in glob_opt.lower()) and ("hopping" in glob_opt.lower()):
            args = {'bounds': bnds, 'tol': tol}
            global_results = opt.basinhopping(get_trng_rms, initial_param_guess, niter=100, minimizer_kwargs=args,                                             stepsize=bas_size)
        else:
            print('Global optimizer name not recognized. Please check spelling and try again.')
            sys.exit()
        elapsed_time = process_time() - t
        save_solutions(global_results, glob_opt, elapsed_time)
        mod_precip_glob, obs_precip_glob = prcp.get_opt_precip_arrays(data, global_results.x, trng_start_date, vld_end_date)
        save_vals(glob_opt, mod_precip_glob, obs_precip_glob, trng_start_date, vld_end_date, mod_file_name, obs_file_name)
        if (loc_opt.lower() == "n/a") or (loc_opt.lower() == "na") or (loc_opt.lower() == "n a"):
            return global_results
        t_2 = process_time()
        if (loc_opt.lower() == "nelder mead") or (loc_opt.lower() == "nm") or (loc_opt.lower() == "neldermead"):
            local_results = opt.minimize(get_trng_rms, global_results.x, method='Nelder-Mead', tol=tol)
        elif ("powell" in loc_opt.lower()):
            local_results = opt.minimize(get_trng_rms, global_results.x, method='Powell', tol=tol)
        else:
            print('Local optimizer name not recognized. Please check spelling and try again.')
            sys.exit()
        elapsed_time_2 = process_time() - t_2
        save_solutions(local_results, loc_opt, elapsed_time_2)
        mod_precip_loc, obs_precip_loc = prcp.get_opt_precip_arrays(data, local_results.x, trng_start_date, vld_end_date)
        save_vals(loc_opt, mod_precip_loc, obs_precip_loc, trng_start_date, vld_end_date, mod_file_name, obs_file_name)
        return local_results

    def save_vals(optimizer, calc_array, obs_array, start, end, calc_file_name, obs_file_name):
        if not os.path.exists(irrig_fldr):
            os.mkdir(irrig_fldr)
        dailydatelist = pd.date_range(start, end).to_pydatetime()
        rms_df = pd.DataFrame(rms_list)
        params_df = pd.DataFrame(params_list)
        obs_precip_df = pd.DataFrame(obs_array, index=dailydatelist, columns=data.columns.values)
        calc_precip_df = pd.DataFrame(calc_array, index=dailydatelist, columns=data.columns.values)
        rms_df.to_csv('{0}/{1}_rms_vals.csv'.format(irrig_fldr, optimizer), header=None)
        obs_precip_df.to_csv(obs_file_name,
                             header=data.columns.values)
        calc_precip_df.to_csv(calc_file_name,
                              header=data.columns.values)
        params_df.to_csv('{0}/{1}_parameter_vals.csv'.format(irrig_fldr, optimizer), header=None)

    def save_solutions(res, optimizer, tottime, **size):
        if not os.path.exists(irrig_fldr):
            os.mkdir(irrig_fldr)
        txt_file_name = '{0}/{1}_solutions.txt'.format(irrig_fldr, optimizer)
        solutions = '  '.join([str(x) for x in res.x])
        s = optimizer
        s += " " + str(today)
        s += '\nSolutions (Z*, Ks, Lambda, Kc): ' + solutions
        message = '  '.join([str(x) for x in res.message])
        s += '\nMessage: ' + message
        s += '\nFunction objective value: ' + str(res.fun)
        s += '\nNumber of evaluations: ' + str(res.nfev)
        s += '\nNumber of iterations: ' + str(res.nit)
        if (optimizer.lower() == 'basinhopping') or (optimizer.lower() == 'basin-hopping') or (
                optimizer.lower() == 'basin hopping'):
            s += '\nBasin size: ' + str(size)
        s += '\nTime to run (in seconds): ' + str(tottime)
        with open(txt_file_name, 'w') as text_file:
            text_file.write(s)
        print(s)

    def get_validation_rms(params):
        model_precip, obs_precip = prcp.get_opt_precip_arrays(data, params, vld_start_date, vld_end_date)
        rms = np.sqrt(mean_squared_error(model_precip, obs_precip))
        print('validation data rms: ' + str(rms))
        save_vals("validation", model_precip, obs_precip, vld_start_date, vld_end_date)
        with open("validation_rms", 'w') as text_file:
            text_file.write("Validation data root mean square error:  " + str(rms))

    # results = run_selected_optimizers(optimizer1, optimizer2)
    # params = results.x
    params = [30.23291347240082, 15.342681210842358, 4.997641994001076, 3.0607181991905216e-05]
    params = [30.204627556737574, 37.981574949380935, 1.4340021218801704, 0.009608286115623779]

    p_calc_array, p_obs_array = prcp.get_opt_precip_arrays(data, params, trng_start_date, vld_end_date)
    p_calc_array = np.array(p_calc_array)
    p_obs_array = np.array(p_obs_array)
    p_calc_array = p_calc_array.transpose()
    p_obs_array = p_obs_array.transpose()
    save_vals(optimizer, p_calc_array, p_obs_array, trng_start_date, vld_end_date, mod_file_name, obs_file_name)

    monthly_irrig_df = irrig.get_monthly_irrig_df(data, params, trng_first_date, vld_last_date)
    monthly_irrig_df.to_csv(irrig_csv)
    # # get_validation_rms(params)
    daily_irrig_df = irrig.get_daily_irrig_df(data, params, trng_first_date, vld_last_date)

    daily_irrig_df.to_csv(irrig_fldr + "/" + optimizer1 + "_daily_irrig_outputs.csv")
    return params
    # return results.x



if __name__ == '__main__':
    in_file_name = 'Model_Setup.txt'
    gldas_data = rdr.read_model_setup_txt(in_file_name)
    gldas_file_path, out_id, num_cols, num_rows, trng_date_1, trng_date_2, vld_date_1, vld_date_2, opt_1, basin_size, \
    opt_2, irrig_out_folder, vis_out_folder, visualization_array = gldas_data
    if opt_2.lower() == "na" or (opt_2.lower() == "n/a") or (opt_2.lower() == "n a"):
        optimizer = opt_1
    else:
        optimizer = opt_1 + '_' + opt_2
    irrig_id_out_folder = irrig_out_folder + "/" + out_id
    obs_precip_name = irrig_id_out_folder + '/observed_precip.csv'
    mod_precip_name = irrig_id_out_folder + '/' + optimizer + '_modeled_precip.csv'
    irrig_out_csv = irrig_id_out_folder + '/' + optimizer + "_monthly_irrigation_outputs.csv"
    trng_first_date = pd.to_datetime(trng_date_1)
    trng_last_date = pd.to_datetime(trng_date_2)
    vld_first_date = pd.to_datetime(vld_date_1)
    vld_last_date = pd.to_datetime(vld_date_2)
    if os.path.exists(mod_precip_name) and os.path.exists(irrig_out_csv):
        print("Irrigation optimization already completed; creating visualization files")
        # skip all these steps and use visualization stuff only
    else:
        temp = rdr.get_gldas_df('air_temp_avg', gldas_file_path) - 273.15  # GLDAS temperature in Kelvin, converting to Celsius
        precip = rdr.get_gldas_df('total_precipitation', gldas_file_path)  # GLDAS precip in kg/m^2*day, which is equivalent to mm/day
        sm = rdr.get_gldas_df('top_soil_moisture', gldas_file_path)/100  # GLDAS soil moisture in kg/m^2, to m^3/m^3 by dividing by layer thickness (100 mm)
        data = pd.concat([temp, precip, sm, sm.diff()], keys=['temp', 'precip', 'sm', 'sm_delta'], names=['source', 'date'])
        if data.shape[1] != num_cols * num_rows:
            print("Specified latitude and longitude range do not correspond to data. Please check files.")
            sys.exit()
        data.reset_index(inplace=True)
        data.set_index(['date', 'source'], inplace=True)
        data.dropna()
        data.sort_index(inplace=True)
        if not os.path.exists(irrig_id_out_folder):
            os.makedirs(irrig_id_out_folder)
        data.to_csv(os.path.join(irrig_id_out_folder, "raw_data.csv"))
        results = find_optimal_parameters(data, trng_first_date, trng_last_date, vld_first_date, vld_last_date, opt_1,
                                          basin_size, opt_2, out_id, irrig_out_csv, obs_precip_name, mod_precip_name)
        params = results.x



    # visualization outputs

    trng_temporal_error_vis, vldtn_temporal_error_vis, t_v_temporal_error_vis, trng_geo_error_vis, \
    vldtn_geo_error_vis, t_v_geo_error_vis, mean_irrig, mean_irrig_vs_precip, mean_obs_precip, mean_obs_vs_mod_precip, \
    obs_vs_mod_cell, o_m_cell_list, irrig_vs_precip_cell, i_p_cell_list = visualization_array



    if trng_temporal_error_vis:
        epg.get_temporal_error_plot(trng_first_date, trng_last_date, obs_precip_name, mod_precip_name, irrig_id_out_folder,
                                    vis_out_folder, out_id)
    if vldtn_temporal_error_vis:
        epg.get_temporal_error_plot(vld_first_date, vld_last_date, obs_precip_name, mod_precip_name, irrig_id_out_folder,
                                    vis_out_folder, out_id)
    if t_v_temporal_error_vis:
        epg.get_temporal_error_plot(trng_first_date, vld_last_date, obs_precip_name, mod_precip_name, irrig_id_out_folder,
                                    vis_out_folder, out_id)
    if trng_geo_error_vis:
        epg.get_spatial_error_plot(trng_first_date, trng_last_date, obs_precip_name, mod_precip_name, num_rows,
                                   num_cols, irrig_id_out_folder, vis_out_folder, out_id)
    if vldtn_geo_error_vis:
        epg.get_spatial_error_plot(vld_first_date, vld_last_date, obs_precip_name, mod_precip_name, num_rows,
                                   num_cols, irrig_id_out_folder, vis_out_folder, out_id)
    if t_v_geo_error_vis:
        epg.get_spatial_error_plot(trng_first_date, vld_last_date, obs_precip_name, mod_precip_name, num_rows,
                                   num_cols, irrig_id_out_folder, vis_out_folder, out_id)
    if mean_irrig:
        opg.plot_mean_irrigation(trng_first_date, vld_last_date, irrig_out_csv, vis_out_folder, out_id, obs='na')
    if mean_irrig_vs_precip:
        opg.plot_mean_irrigation(trng_first_date, vld_last_date, irrig_out_csv, vis_out_folder, out_id, obs=obs_precip_name)
    if mean_obs_precip:
        opg.display_and_save_obs_precip('m', trng_first_date, vld_last_date, vis_out_folder, obs_precip_name, out_id)
    if mean_obs_vs_mod_precip:
        opg.plot_obs_mod_avg(trng_first_date, vld_last_date, vis_out_folder, obs_precip_name, mod_precip_name, out_id)
    if obs_vs_mod_cell:
        for coords in o_m_cell_list:
            opg.plot_obs_mod_cell(trng_first_date, vld_last_date, vis_out_folder, obs_precip_name, mod_precip_name,
                                  num_cols, num_rows, coords, out_id)
    if irrig_vs_precip_cell:
        for coords in i_p_cell_list:
            opg.plot_irrig_at_cell(trng_first_date, vld_last_date, vis_out_folder, irrig_out_csv, obs_precip_name,
                                  num_cols, num_rows, coords, out_id)


