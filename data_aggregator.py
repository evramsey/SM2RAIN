import os
import pandas as pd

rootInputDir = "/Users/lizramsey/PycharmProjects/Syria/GLDAS_Reader/GLDAS_output/Daily/"

def compileThoseDates(input_name, output_name):
    the_file = rootInputDir + "/" + input_name
    dir = os.listdir(the_file)
    df = pd.DataFrame()
    for dirfile in dir:
        if dirfile != ".DS_Store":
            print(dirfile)
            dirName = the_file + "/" + dirfile
            rdr = pd.read_csv(dirName, index_col=0, parse_dates=["Date"])
            df = df.append(rdr)
    df.sort_index(inplace=True)
    print(df)
    df.to_csv(the_file + "/" + output_name + ".csv")

#TODO: run for all variables required
compileThoseDates("Top_Soil_Moisture", "Top_SM_Composite")