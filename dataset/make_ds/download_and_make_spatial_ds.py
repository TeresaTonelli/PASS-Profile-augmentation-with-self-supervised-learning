import os
import random
import torch
import numpy as np
import xarray as xr
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


def from_day_rad_to_months_day(day_rad):
    day_total = (day_rad * 365) / (2 * np.pi)
    month = day_total // 30
    day = day_total % 30
    if day == 0.0:
        month = month -1
        day = 30
    return int(month + 1), int(day)


def extract_data_time(dataset_tuple):
    ds = pd.read_csv(dataset_tuple, header=None)
    dates = []
    for j_1 in range(1, ds.shape[1]):
        sample_tuple = ds.iloc[:, j_1]
        year = int([float(sample_tuple.iloc[1].strip()) for sample_tuple.iloc[1] in sample_tuple.iloc[1].strip("[]").split(",")][0])         
        day_rad = [float(sample_tuple.iloc[2].strip()) for sample_tuple.iloc[2] in sample_tuple.iloc[2].strip("[]").split(",")][0]
        month, day = from_day_rad_to_months_day(day_rad)
        start_date = date(year, month, day)
        dates.append(start_date.strftime('%Y-%m-%d'))
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
        if os.listdir(out_chl_directory_i):
            continue
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



def pypeline_download_copernicus_dataset(dataset_tuple, list_copernicus_data_directories, list_vars, years_interval, list_var_folders, surface_depth, list_time_resolution):
    #list_time = built_in_generate_datetime([years_interval[0], years_interval[1]])
    list_time = extract_data_time(dataset_tuple)
    print("len list time", len(list_time))
    print("list time", list_time[:8])
    for i in range(len(list_copernicus_data_directories)):
        copernicus_data_directory = list_copernicus_data_directories[i]
        print("copernicus dataset", copernicus_data_directory)
        out_directory = "/leonardo_scratch/large/userexternal/ttonelli/OGS/" + list_var_folders[i]   
        time_step = list_time_resolution[i]
        download_copernicus_dataset(copernicus_data_directory, list_vars, list_time, out_directory, surface_depth, time_step) 


Copernicus_download = False
if Copernicus_download == True:
    dataset_tuple = "/leonardo_work/OGS23_PRACE_IT_0/ttonelli/SSint_profiles/dataset/NITRATE/complete_test_mnt.csv"
    copernicus_data_directory = ["cmems_mod_med_phy-cur_my_4.2km_P1D-m"]
    list_vars = ["vo", "uo"]    #da chiedere a GC se ne devo aggiungere altre
    years_interval = [2014, 2021]   #Anche qua si può cambiare, in base a quanti dati ho
    list_var_folders = ["currents_Copernicus_data"]
    surface_depth = 1.0182366371154785
    list_time_resolution = ["daily"]   
    pypeline_download_copernicus_dataset(dataset_tuple, copernicus_data_directory, list_vars, years_interval, list_var_folders, surface_depth, list_time_resolution)



#Transform netCDF file into csv file part

def make_spatial_ds(spatial_nc_file, dataset_tuple):  
    ds_uv = xr.open_dataset(spatial_nc_file)
    ds = pd.read_csv(dataset_tuple, header=None)   #QUESTO TUTTO DA CONTROLLARE
    m_1 = ds.shape[1]    #m should detect the number of columns
    m_2 = ds_uv.shape[1]
    lats_uv = ds_uv["lat"].values      #Non è detto che lat sia la chiave corretta, CHECK
    lons_uv = ds_uv["lon"].values
    times_uv = ds_uv["time"].values    #Con values --> dovrebbe venire un np.array, non un xrarray
    U_uv = ds["u"].values
    V_uv = ds["v"].values
    #ALTERNATIVE: 
    #for i in range(len(lats)):
    #    for j in range(len(lons)):
    #        value = ds["temperature"].isel(lat=i, lon=j).values
    assert len(lats_uv) == len(lons_uv) == len(times_uv)
    assert len(lats_uv) == m_2
    D3 = list()     #forse è meglio un altro tipo, per ora la lascio come una lista
    for j_1 in range(m_1):      #for d1 in D1
        dj = ds.iloc[:, j_1]
        year = dj.iloc[0][0]     #da capire se è giusto l'indice della riga, forse devo aumentare di 1 per il nome dei float --> CHECK
        day_rad = dj.iloc[1][0]
        dt = dj.iloc[1][1] - dj.iloc[1][0]    #qua serve sicuramente la funzione che li trasforma in numeri interi
        U = V = np.zeros(7)
        lat_1 = ds.iloc[2, j_1]
        lon_1 = ds.iloc[3, j_1]
        assert len(lat_1) == len(lon_1)
        assert len(lat_1) == 7
        for i in range(len(lat_1)):
            for j_2 in range(m_2):   #for d2 in D2
                lat_2 = lats_uv[j_2]
                lon_2 = lons_uv[j_2]
                year_2 = int(times_uv[j_2][:4])
                day_rad_2 = ((int(times_uv[j_2][4:6]) - 1) * 30 + int(times_uv[j_2][6:8])) * 2 * np.pi / 365
                #CHECK
                check = lat_1 == lat_2 and lon_1 == lon_2 and year == year_2 and day_rad == day_rad_2
                if check == True:
                    U[i] = U_uv[j_2]
                    V[i] = V_uv[j_2]
                else:
                    continue
        d3 = [lat_1, lon_1, U, V, dt]
        D3.append(d3)
    assert len(D3) == ds.shape[1]
    D3.to_csv("/leonardo_work/OGS23_PRACE_IT_0/ttonelli/SSint_profiles/dataset/currents_data_tuple.csv")  #da controllare sta forma
    return None


def create_currents_dataset(nc_directory, dataset_tuple): 
    ds = pd.read_csv(dataset_tuple, header=None)
    for j_1 in range(1, ds.shape[1]):
        sample_tuple = ds.iloc[:, j_1]
        year = int([float(sample_tuple.iloc[1].strip()) for sample_tuple.iloc[1] in sample_tuple.iloc[1].strip("[]").split(",")][0])         
        day_rad = [float(sample_tuple.iloc[2].strip()) for sample_tuple.iloc[2] in sample_tuple.iloc[2].strip("[]").split(",")][0]
        month, day = from_day_rad_to_months_day(day_rad)
        nc_file = xr.open_dataset(os.path.makedirs(nc_directory, str(year), str(month), str(day)))
            



