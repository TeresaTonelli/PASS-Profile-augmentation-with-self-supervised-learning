import os
import random
import torch
import numpy as np
import pandas as pd
import netCDF4 as nc

import copernicusmarine
from datetime import date, timedelta

import sys
sys.path.append("../../.")   

from dataset.make_ds.utils import shuffle_dict
from preprocessing import *


#DOWNLOAD from Copernicus part

def built_in_generate_datetime(year_intervals):
    start_date = date(year_intervals[0], 1, 1)
    end_date = date(year_intervals[1], 12, 31)
    delta = timedelta(days=1)
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current.strftime('%Y-%m-%d'))
        current += delta
    dates = [d + 'T00:00:00' for d in dates]
    return dates


def download_copernicus_dataset(copernicus_data_directory, var_list, list_times, out_chl_directory, surface_depth=0, time_step="monthly"):
    for i in range(len(list_times)):     
        out_chl_directory_i = out_chl_directory + "/" + str(list_times[i][:4])
        out_chl_directory_i = out_chl_directory_i + "/" + str(list_times[i][5:7])
        if time_step == "daily":
            out_chl_directory_i = out_chl_directory_i + "/" + str(list_times[i][8:10])
        if not os.path.exists(out_chl_directory_i):
            os.makedirs(out_chl_directory_i)
        copernicusmarine.subset(
        dataset_id=copernicus_data_directory,
        variables=var_list,
        minimum_longitude=-6,
        maximum_longitude=36.29166793823242,
        minimum_latitude=30.1875,
        maximum_latitude=45.97916793823242,
        start_datetime=list_times[i],
        end_datetime=list_times[i],
        minimum_depth=surface_depth,    #nel dataset Copenicus è 1.0182366371154785
        maximum_depth=surface_depth,
        output_directory=out_chl_directory_i
        )
    return None



def pypeline_download_copernicus_dataset(list_copernicus_data_directories, list_vars, years_interval, list_var_folders, surface_depth, list_time_resolution):
    list_time = built_in_generate_datetime([years_interval[0], years_interval[1]])
    for i in range(len(list_copernicus_data_directories)):
        copernicus_data_directory = list_copernicus_data_directories[i]
        out_directory = "/leonardo_scratch/large/userexternal/ttonelli/OGS/" + list_var_folders[i]   
        time_step = list_time_resolution[i]
        download_copernicus_dataset(copernicus_data_directory, list_vars, list_time, out_directory, surface_depth, time_step) 


Copernicus_download = True
if Copernicus_download == True:
    copernicus_data_directory = "cmems_mod_med_phy-cur_my_4.2km_P1D-m"
    list_vars = ["vo, uo"]    #da chiedere a GC se ne devo aggiungere altre
    years_interval = [2015, 2021]   #Anche qua si può cambiare, in base a quanti dati ho
    list_var_folders = ["currents_Copernicus_data"]
    surface_depth = 1.0182366371154785
    list_time_resolution = ["daily"]   
    pypeline_download_copernicus_dataset(copernicus_data_directory, list_vars, years_interval, list_var_folders, surface_depth, list_time_resolution)



#Transform netCDF file into csv file part

def make_spatial_ds(spatial_nc_file):  
    pass

