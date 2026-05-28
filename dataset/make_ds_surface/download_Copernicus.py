from importlib.resources import files
import os
import numpy as np
import xarray as xr
import pandas as pd
import netCDF4 as nc
from itertools import islice

import copernicusmarine
from copernicusmarine.core_functions.exceptions import CoordinatesOutOfDatasetBounds
from datetime import date, timedelta

import sys
sys.path.append("../../.")   


#DOWNLOAD from Copernicus part


def csv_array_correction(dataset_csv):
    df = pd.read_csv(dataset_csv, header=None)
    print("df shape", df.shape)
    var_arrays = df.iloc[-1, :].to_list()
    ##print("var_arrays", var_arrays)
    print("var arrays len", len(var_arrays))
    #print("var arrays [1]", var_arrays[1])
    assert type(var_arrays[10]) == str
    #var_arrays = [v.replace("array([", "").replace("], dtype=float32)", "").replace("\n", "").split(",") for v in var_arrays[1:]]
    var_arrays = var_arrays.replace("array([", "").replace("], dtype=float32)", "").replace("\n", "").split(",") 
    print("var arrays", var_arrays)
    print("var arrays len", len(var_arrays))
    print("var arrays [0]", var_arrays[0])
    var_arrays = [var_arrays[0]] + [v.replace("array([", "").replace("], dtype=float32)", "").replace("\n", "").split(",") for v in var_arrays[1:]]
    print("var arrays", var_arrays)
    df.iloc[-1, :] = var_arrays
    #df.to_csv(dataset_csv)


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
    day_total = round((day_rad * 365) / (2 * np.pi))
    month = day_total // 30
    day = day_total % 30
    if month == 12:
        month = month-1
        day = day + 30
    if day == 0.0 or day==0:
        month = month -1
        day = 30
    return int(month + 1), int(day)


def extract_data_time(dataset_csv):
    ds = pd.read_csv(dataset_csv, header=None)
    n_vars, n = ds.shape
    dates = []
    for j_1 in range(1, ds.shape[1]):
        sample = ds.iloc[:, j_1]
        year_tuple = [int(float(y)) for y in sample.iloc[1].strip('[]').split(',')]
        day_rad_tuple = [float(d) for d in sample.iloc[2].strip('[]').split(',')]
        month_day_tuple = [from_day_rad_to_months_day(day_rad) for day_rad in day_rad_tuple]
        start_date_tuple = [date(year, month, day).strftime('%Y-%m-%d') + 'T00:00:00' for year, (month, day) in zip(year_tuple, month_day_tuple)]
        dates.append(start_date_tuple)
    assert len(dates) == n-1
    return dates


def extract_lats_lons(dataset_tuple):
    ds = pd.read_csv(dataset_tuple, header=None)
    lons = []
    lats = []
    for j_1 in range(1, ds.shape[1]):
        sample_tuple = ds.iloc[:, j_1]
        lat_tuple = [float(l_a) for l_a in sample_tuple.iloc[3].strip('[]').split(',')]
        lon_tuple = [float(l_o) for l_o in sample_tuple.iloc[4].strip('[]').split(',')]
        lons.append(lon_tuple)
        lats.append(lat_tuple)
    assert len(lats) == len(lons) == ds.shape[1]-1
    return lons, lats


def download_copernicus_dataset(copernicus_data_directory, var, list_lons, list_lats, list_times, out_chl_directory, max_depth=0.0, epsilon=0.05):
    chla_maps = []
    for i in range(len(list_times)): 
        out_chl_directory_i = "/".join([out_chl_directory, list_times[i][:4], list_times[i][5:7], list_times[i][8:10]])
        os.makedirs(out_chl_directory_i, exist_ok=True)
        #if len([f for f in os.listdir(out_chl_directory_i) if (f.endswith(".nc") and str(round(float(list_lons[i] - epsilon), 2)) in f and str(round(float(list_lats[i]+epsilon),2)) in f)]) > 0:
        #    print("f", [f for f in os.listdir(out_chl_directory_i) if (f.endswith(".nc") and str(abs(int(list_lons[i]))) in f and str(abs(int(list_lats[i]))) in f)][0])
        #    continue
        try:
            chla_cop = copernicusmarine.subset(
            dataset_id=copernicus_data_directory,
            variables=[var],
            minimum_longitude=list_lons[i] - epsilon,
            maximum_longitude=list_lons[i] + epsilon,
            minimum_latitude=list_lats[i] - epsilon,
            maximum_latitude=list_lats[i] + epsilon,
            start_datetime=list_times[i],
            end_datetime=list_times[i],
            minimum_depth=0,    
            maximum_depth=max_depth,
            output_directory=out_chl_directory_i
            )
            print("chla cop", chla_cop.file_path)
            print("lat, lon str", list_lons[i], list_lats[i])
            try:
                ds_chl = nc.Dataset(os.path.join(chla_cop.file_path)) #[f for f in os.listdir(out_chl_directory_i) if (f.endswith(".nc") and str(abs(int(list_lons[i]))) in f and str(abs(int(list_lats[i]))) in f)][0]))
                #print("file f", [f for f in os.listdir(out_chl_directory_i) if (f.endswith(".nc") and str(round(float(list_lons[i] - epsilon), 2))in f and str(round(float(list_lats[i]+epsilon),2)) in f)][0])
                #ds_chl = nc.Dataset(os.path.join(out_chl_directory_i,[f for f in os.listdir(out_chl_directory_i) if (f.endswith(".nc") and str(round(float(list_lons[i] - epsilon), 2))in f and str(round(float(list_lats[i]+epsilon),2)) in f)][0]))
                chl = np.squeeze(np.array(ds_chl.variables[var][:]))
                m = min(7, chl.shape[0], chl.shape[1])
                chla_maps.append(chl[:m, :m])

            except Exception as e:
                print("Exception loading")
                chla_maps.append(np.zeros([7, 7]))
        except CoordinatesOutOfDatasetBounds as e: 
            print("Exception download coordinates")
            chla_maps.append(np.zeros([7, 7]))
    assert len(chla_maps) == len(list_lats)
    return chla_maps



def pypeline_download_copernicus_dataset(dataset_csv, copernicus_data_directory, var, var_folder, index_start, index_end, max_depth=0.0, epsilon=0.05):
    list_time = extract_data_time(dataset_csv)[index_start:index_end]
    list_lons, list_lats = extract_lats_lons(dataset_csv)
    list_lons = list_lons[index_start:index_end]
    list_lats = list_lats[index_start:index_end]
    out_directory = os.path.join("/leonardo_scratch/large/userexternal/ttonelli/OGS/", var_folder)
    list_chla = []
    for k in range(len(list_time)):
        chla = download_copernicus_dataset(copernicus_data_directory, var, list_lons[k], list_lats[k], list_time[k], out_directory, max_depth, epsilon) 
        chla = [c.flatten() for c in chla]
        list_chla.append(chla)
    ds = pd.read_csv(dataset_csv, header=None)
    df = pd.DataFrame({
        "year": [ds.iloc[:, l].iloc[1].strip() for l in range(index_start+1, index_end+1)],     #range(1, ds.shape[1])],
        "day_rad": [ds.iloc[:, l].iloc[2].strip() for l in range(index_start+1, index_end+1)],    #range(1, ds.shape[1])],
        "lat": [ds.iloc[:, l].iloc[3].strip() for l in range(index_start+1, index_end+1)], #list_lats,
        "lon": [ds.iloc[:, l].iloc[4].strip() for l in range(index_start+1, index_end+1)],  #list_lons,
        "chla": [ds.iloc[:, l].iloc[-1].strip() for l in range(index_start+1, index_end+1)],
        "chla_surface": list_chla  #[c.flatten() for c in chla]
    })
    df_T = df.T
    df_T.columns = [f"sample_{i}" for i in range(df_T.shape[1])]
    t = "train" if "train" in str(dataset_csv) else "test"  
    df_T.to_csv(os.path.join("/leonardo_work/OGS23_PRACE_IT_0/ttonelli/SSint_profiles/dataset", "CHLA", f"complete_{t}_mnt_surf.csv"))


Copernicus_download = False
if Copernicus_download == True:

    dataset_csv = "/leonardo_work/OGS23_PRACE_IT_0/ttonelli/SSint_profiles/dataset/CHLA/complete_test_mnt.csv"
    m = pd.read_csv(dataset_csv, header=None).shape[1]
    copernicus_data_directory = "cmems_obs-oc_med_bgc-plankton_my_l4-gapfree-multi-1km_P1D"
    var = "CHL"    
    var_folder = "SSint_surface_chl"
    pypeline_download_copernicus_dataset(dataset_csv, copernicus_data_directory, var, var_folder, 0, m-1)



Csv_correction = True
if Csv_correction == True:
    dataset_csv = "/leonardo_work/OGS23_PRACE_IT_0/ttonelli/SSint_profiles/dataset/CHLA/complete_test_mnt_surf_cp.csv"
    csv_array_correction(dataset_csv)