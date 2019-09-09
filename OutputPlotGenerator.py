import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import os, sys

def get_monthly_cell_precip(cell_num, precip_file):
    precip_df = pd.read_csv(precip_file, parse_dates=True, index_col=0)
    precip_vals = precip_df.groupby(pd.Grouper(freq='MS')).sum()
    precip_val = precip_vals.iloc[:,cell_num]
    return precip_val

'''Plot line graph of average observed precipitation for all cells for either
monthly or annual resolution'''
def plot_avg_obs_precip(temp_res, start, end, obs_precip):
    if temp_res.lower() == 'month' or temp_res.lower() == 'm':
        frequency = 'MS'
    elif temp_res.lower() == 'year' or temp_res.lower() == 'y':
        frequency = 'Y'
    else:
        print('Check temporal resolution in Average Observed Basin Precipitation function')
        sys.exit()
    date_list = pd.date_range(start, end, freq=frequency)
    precip = pd.read_csv(obs_precip, parse_dates=True, index_col=0)
    avgPrecip = precip.groupby(pd.Grouper(freq=frequency)).sum().mean(axis=1)
    avgPrecip = avgPrecip.drop(avgPrecip.index[0])
    plt.plot(date_list, avgPrecip, color='#0101DF')

def display_and_save_obs_precip(temp_res, start, end, vis_out, obs_precip, out_id):
    plot_avg_obs_precip(temp_res, start, end, obs_precip)
    plt.xlabel("Time")
    plt.ylabel("Precipitation (mm/{0})".format(temp_res.lower()))
    plt.savefig(vis_out + "/" + out_id + "/average_obs_basin_precip.png")
    plt.grid(b=None)
    plt.show()

def get_df(irrigFile):
    inFile = os.path.join(irrigFile)
    inFile = pd.read_csv(inFile, index_col=0, parse_dates=True, index="date")
    return inFile

def plot_mean_irrigation(start_date, end_date, irrig_file, vis_file, out_id, obs):
    if obs != 'na':
        plot_avg_obs_precip('m', start_date, end_date, obs)
        obs_line = mlines.Line2D([], [], color='#0101DF', label="Observed Precipitation")
    vis_file = vis_file + '/' + out_id
    monthlydatelist = pd.date_range(start_date, end_date, freq='MS')
    irrig_df = pd.read_csv(irrig_file, index_col=0, parse_dates=True)
    mean_irrig = irrig_df.mean(axis=1)
    mean_irrig = mean_irrig.drop(mean_irrig.index[0])
    print(len(mean_irrig), len(monthlydatelist))
    plt.plot(monthlydatelist, mean_irrig, "#4FD093")
    model_line = mlines.Line2D([], [], color='#4FD093', label="Modeled Irrigation")
    if obs != 'na':
        plt.legend(handles=[model_line, obs_line])
    else:
        plt.legend(handles=[model_line])
    plt.xlabel('Time')
    plt.ylabel('mm/month')
    plt.grid(b=None)
    if obs != 'na':
        plt.savefig(os.path.join(vis_file, "mean_irrigation_vs_obs.png"))
    else:
        plt.savefig(os.path.join(vis_file, "mean_irrigation.png"))
    plt.show()

def get_cell_num(i, j, num_cols, num_rows):
    cell_num = j + (i * num_cols)
    if (i > num_rows - 1) or (j > num_cols - 1):
        print("At least one cell (i,j) does not exist. Please check numbers and try again.")
        sys.exit()
    else:
        return cell_num

def plot_obs_mod_avg(start_date, end_date, vis_folder, obs_precip, mod_precip, out_id):
    plot_avg_obs_precip('month', start_date, end_date, obs_precip)
    date_list = pd.date_range(start_date, end_date, freq='MS')
    m_precip = pd.read_csv(mod_precip, parse_dates=True, index_col=0)
    avg_mod_precip = m_precip.groupby(pd.Grouper(freq='MS')).sum().mean(axis=1)
    avg_mod_precip = avg_mod_precip.drop(avg_mod_precip.index[0])
    plt.plot(date_list, avg_mod_precip, color='#22d0f2')
    obs_line = mlines.Line2D([], [], color='#0101DF', label="Observed Precipitation")
    mod_line = mlines.Line2D([], [], color='#22d0f2', label="Modeled Precipitation")
    plt.legend(handles=[mod_line, obs_line])
    vis_folder = vis_folder + '/' + out_id
    plt.savefig(os.path.join(vis_folder, "mean_obs_vs_mod_precip.png"))
    plt.show()

def plot_obs_mod_cell(start_date, end_date, vis_folder, obs_precip, mod_precip, num_cols, num_rows, coords, out_id):
    if len(coords) != 2:
        print("Coordinates for modeled vs. observed precipitation incorrect. Please check that coordinates are in (i,j) "
              "format")
        return
    cell_num = get_cell_num(coords[0], coords[1], num_cols, num_rows)
    obs_df = get_monthly_cell_precip(cell_num, obs_precip)
    mod_df = get_monthly_cell_precip(cell_num, mod_precip)
    monthly_date_list = pd.date_range(start_date, end_date, freq='MS')
    plt.plot(monthly_date_list, obs_df, '#0101DF', monthly_date_list, mod_df, '#9FE4F1')
    obs_line = mlines.Line2D([], [], color='#0101DF', label="Observed Precipitation")
    model_line = mlines.Line2D([], [], color='#9FE4F1', label="Modeled Precipitation")
    plt_name = 'obs_vs_mod_precip_cell_' + str(coords[0]) + '_' + str(coords[1]) + '.png'
    plt.legend(handles=[model_line, obs_line])
    plt.xlabel('Time')
    plt.ylabel('mm/month')
    plt.grid(b=None)
    vis_folder = vis_folder + '/' + out_id
    plt.savefig(os.path.join(vis_folder, plt_name))
    plt.show()

def plot_irrig_at_cell(start_date, end_date, vis_folder, irrig_file, obs_precip, num_cols, num_rows, coords, out_id):
    if len(coords) != 2:
        print("Coordinates for irrigation and precipitation plot are incorrect. Please check that coordinates are in "
              "(i,j) format")
        return
    cell_num = get_cell_num(coords[0], coords[1], num_cols, num_rows)
    irrig_df = pd.read_csv(irrig_file, index_col=0, parse_dates=True)
    irrig_vals = irrig_df.iloc[:,cell_num]
    precip = get_monthly_cell_precip(cell_num, obs_precip)
    irrig_vals = irrig_vals.drop(irrig_vals.index[0])
    precip = precip.drop(precip.index[0])
    precip_line = mlines.Line2D([], [], color='#0101DF', label="Observed Precipitation")
    irrig_line = mlines.Line2D([], [], color='#44A720', label="Irrigation")
    monthlydatelist = pd.date_range(start_date, end_date, freq='M')
    plt.plot(monthlydatelist, precip, '#0101DF', monthlydatelist, irrig_vals, '#44A720')
    plt_name = 'irrig_vs_precip_cell_' + str(coords[0]) + str(coords[1]) + '.png'
    plt.legend(handles=[irrig_line, precip_line])
    plt.xlabel('Time')
    plt.ylabel('mm/month')
    plt.grid(b=None)
    vis_folder = vis_folder + '/' + out_id
    plt.savefig(os.path.join(vis_folder, plt_name))
    plt.show()

