import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns; sns.set()

'''Calculate squared differences between observed and modeled precipitation for each cell and day'''
def get_error(filename, obs_precip, mod_precip):
    my_date_parser = lambda x: pd.datetime.strptime(x, "%Y-%m-%d")
    obs = pd.read_csv(obs_precip, index_col=0, parse_dates=True, date_parser=my_date_parser)
    mod = pd.read_csv(mod_precip, index_col=0, parse_dates=True, date_parser=my_date_parser)
    error = (obs-mod)**2
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    error.to_csv(filename)
    return error

'''Group data by month'''
def get_monthly_plot_data(errorDF, start_date, end_date):
    plot_data = []
    errorDF.dropna()
    errorDF = errorDF.groupby(pd.Grouper(freq='M'))
    for date, val in errorDF:
        if (date > start_date or date == start_date) and (date < end_date or date == end_date):
            val_list = val.values.flatten()
            plot_data.append(val_list)
    return plot_data

'''Generate and save box plot'''
def create_box_plot(data_to_plot, out_file_name, month_list):
    palette = sns.color_palette("hls", 8)
    fig, ax = plt.subplots()
    flierprops = dict(marker='o', markersize=2, color="#9DE3E5", markeredgecolor="#9DE3E5")
    # ax = sns.boxplot(data=data_to_plot, flierprops=flierprops, palette=palette)
    sns.boxplot(data=data_to_plot, showfliers=False, palette=palette, width=1.2)
    ax.xaxis.set_major_locator(ticker.LinearLocator(len(month_list)))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_ylabel('Error (mm/day)', fontsize=16)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.set(xticklabels=month_list)
    for tick in ax.get_xticklabels():
        tick.set_fontsize(fontsize=10)

    ax.set_xlabel('Time', fontsize=16)

    fig.savefig(out_file_name, bbox_inches='tight')
    plt.show()

'''Either generate square error calculations from observed and modeled precipitation or read them from previously
generated error .csv file'''
def get_error_df(dates, obs_precip, mod_precip, irrig_fldr, out_id):
    error_file_path = irrig_fldr + "/" + out_id + "/error_outputs/" + dates + "_error.csv"
    if os.path.exists(error_file_path):
        my_date_parser = lambda x: pd.datetime.strptime(x, "%Y-%m-%d")
        error_vals = pd.read_csv(error_file_path, index_col=0, parse_dates=True, date_parser=my_date_parser)
    else:
        error_vals = get_error(error_file_path, obs_precip, mod_precip)
    return error_vals

'''Set up irrigation and error output folders'''
def set_up_folders(irrig_folder, vis_file, out_id):
    id_folder = os.path.join(irrig_folder, out_id)
    if not os.path.exists(os.path.join(id_folder, 'error_outputs')):
        os.makedirs(os.path.join(id_folder, 'error_outputs'))
    vis_file_flagged = os.path.join(vis_file, out_id)
    if not os.path.exists(vis_file):
        os.makedir(vis_file)
    if not os.path.exists(vis_file_flagged):
        os.makedir(vis_file_flagged)

'''Generate and save a box plot of square error for each day at each cell across study area'''
def get_temporal_error_plot(start_date, end_date, obs_precip, mod_precip, irrig_file, vis_file, out_flag):
    irrig_folder = os.path.dirname(irrig_file)
    set_up_folders(irrig_folder, vis_file, out_flag)
    date_range = str(start_date.strftime("%m-%Y")) + "_" + str(end_date.strftime("%m-%Y"))
    vis_file_name = vis_file + "/" + out_flag + "/" + date_range + "_temporal_error.png"
    error_vals = get_error_df(date_range, obs_precip, mod_precip, irrig_folder, out_flag)
    monthly_date_list = pd.date_range(start_date, end_date, freq='MS').strftime("%m-%Y")

    yearly_date_list = pd.date_range(start_date, end_date, freq='YS', closed='right').strftime("%m-%Y")
    plot_data = get_monthly_plot_data(error_vals, start_date, end_date)
    create_box_plot(plot_data, vis_file_name, yearly_date_list)

'''Generate and save a plot of mean square error for each cell in data frame'''
def get_spatial_error_plot(start_date, end_date, obs_p, mod_p, n_rows, n_cols, irrig_file, vis_file, out_id):
    irrig_folder = os.path.dirname(irrig_file)
    set_up_folders(irrig_folder, vis_file, out_id)
    date_range = str(start_date.strftime("%m-%Y")) + "_" + str(end_date.strftime("%m-%Y"))
    vis_file_name = vis_file + "/" + out_id + "/" + date_range + "_spatial_error.png"
    error_vals = get_error_df(date_range, obs_p, mod_p, irrig_folder, out_id)
    index_start_names = error_vals[(error_vals.index < start_date)].index
    index_end_names = error_vals[(error_vals.index > end_date)].index
    error_vals.drop(index_start_names, inplace=True, axis=0)
    error_vals.drop(index_end_names, inplace=True)
    mse_vals = error_vals.mean(axis=0)
    error_mtrx = mse_vals.values.reshape(n_rows, n_cols)
    # flip  because seaborn produces a map from top to bottom, but this code calculates bottom to top
    inverse_heat_map = np.flip(error_mtrx, 0)
    colors = ['#C9F0FC', '#A1E0F3', '#79D0EB', '#51C0E2', '#29B0DA', '#01A0D2']
    palette = sns.color_palette(colors)
    ax = sns.heatmap(inverse_heat_map, square=True, cmap=palette, cbar=True, cbar_kws=dict(use_gridspec=False,
                        location="bottom"), xticklabels=False, yticklabels=False)
    plt.savefig(vis_file_name)
    plt.show()

if __name__ == '__main__':
    print("main")