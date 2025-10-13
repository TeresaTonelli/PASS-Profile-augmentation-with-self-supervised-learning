import os
import random
import torch
import numpy as np
import pandas as pd
import netCDF4 as nc

import sys
sys.path.append("../../.")   

from dataset.make_ds.utils import shuffle_dict
from preprocessing import *


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def find_two_nearest(array, value):
    array = np.asarray(array)
    if value <= array[0]:
        return [0, 0]
    if value >= array[-1]:
        return [-1, -1]
    idx = np.searchsorted(array, value)
    return [idx -1, idx]


def find_nearests_interval(array, value, interval):
    array = np.asarray(array)
    list_idx = [i for i in range(len(array)) if np.abs(array[i] - value) < float(interval / 2)]
    return list_idx


def moving_average(x, window_size, mode="same"):
    x = np.asarray(x)
    mov_out = torch.zeros(len(x))
    if mode == "same":
        if window_size > len(x):
            raise ValueError("window_size must be <= length of input array")
        mov_np = np.convolve(x, np.ones(window_size)/window_size, mode=mode)
        for i in range(len(x)):
            mov_out[i] = torch.from_numpy(np.array(mov_np[i]))
        return mov_out   
    elif mode == "valid":
        l = int(window_size / 2)
        for i in range(l):
            mov_out[i] = torch.from_numpy(np.array(x[i]))
        for i in range(len(x)-l, len(x)):
            mov_out[i] = torch.from_numpy(np.array(x[i]))
        if window_size > len(x):
            raise ValueError("window_size must be <= length of input array")
        mov_np = np.convolve(x, np.ones(window_size)/window_size, mode=mode)
        assert len(mov_np) == len(x) - window_size + 1
        for i in range(l, len(x)-l):
            mov_out[i] = torch.from_numpy(np.array(mov_np[i-l]))
        return mov_out 


def discretize(pres, var, max_pres, interval):
    """
    function that take as input a profile, the corresponding pres and create a tensor
    the interval input represents the discretization scale
    the max_pres input represents the maximum pressure considered
    """
    size = int(max_pres / interval)
    discretization_pres = np.arange(0, max_pres, interval)
    out = torch.zeros(size)
    for i in range(size):
        pressure_discretize = discretization_pres[i]
        list_idx = find_nearests_interval(pres, pressure_discretize, interval)
        if len(list_idx) == 0:
            array_idx = find_two_nearest(pres, pressure_discretize)
            if len(array_idx) == 0:
                print("None idx")
                return None
            out[i] = torch.from_numpy((np.array(np.interp(pressure_discretize, pres[array_idx], np.asarray([var[idx] for idx in array_idx])))))
        else: 
            out[i] = torch.from_numpy((np.array([np.mean(np.asarray([var[idx] for idx in list_idx]))])))      
    out = moving_average(out, 5, mode="valid")
    if any(isinstance(el, str) for el in out):
        print("There is a string in the array.")
    return out


def read_date_time(date_time):
    year = int(date_time[0:4])
    month = int(date_time[4:6])
    month = month - 1
    day = int(date_time[6:8])
    day_total = month * 30 + day
    day_rad = day_total * 2 * np.pi / 365
    return year, day_rad


#def make_dict_single_float(path, date_time, variable):
def make_dict_single_float(path, date_time, variable):
    if not os.path.exists(path):
        return None
    year, day_rad = read_date_time(date_time)
    try:
        ds = nc.Dataset(path)  
    except Exception as error:
        print('Caught this error: ' + repr(error))
        return dict(), 0
    lat = ds["LATITUDE"][:].data[0]
    lon = ds["LONGITUDE"][:].data[0]
    psal_df = ds["PSAL"][:].data[:]
    pres_psal_df = ds["PRES_PSAL"][:].data[:]
    temp_df = ds["TEMP"][:].data[:]
    pres_temp_df = ds["PRES_TEMP"][:].data[:]
    if "DOXY" not in ds.variables.keys():
        return dict(), 0
    doxy_df = ds["DOXY"][:].data[:]
    pres_doxy_df = ds["PRES_DOXY"][:].data[:]
    if variable not in ds.variables.keys():
        return dict(), 0
    variable_df = ds[f"{variable}"][:].data[:]
    pres_variable_df = ds[f"PRES_{variable}"][:].data[:]
    if variable == "NITRATE":
        flag_missing_extreme = 1
        flag_counting_measurement = 1
    if variable == "CHLA":
        flag_missing_extreme = missing_extreme_test(pres_variable_df=pres_variable_df, min_extreme=10, max_extreme=180)
        flag_counting_measurement = counting_measurement_test(variable_df=variable_df, tradeoff=50)
    if variable == "BBP700":
        flag_missing_extreme = missing_extreme_test(pres_variable_df=pres_variable_df, min_extreme=30, max_extreme=180)
        flag_counting_measurement = counting_measurement_test(variable_df=variable_df, tradeoff=50)
        variable_df = variable_df * 1000
    max_pres = dict_max_pressure[variable]
    interval = dict_interval[variable]
    temp = discretize(pres_temp_df, temp_df, max_pres, interval)
    psal = discretize(pres_psal_df, psal_df, max_pres, interval)
    doxy = discretize(pres_doxy_df, doxy_df, max_pres, interval)
    variable = discretize(pres_variable_df, variable_df, max_pres, interval)
    name_float = path.split("/")[-1][:-3]
    if temp is None or psal is None or doxy is None or variable is None:
    #if variable is None:
        return dict(), 0
    dict_float = {name_float: [year, day_rad, lat, lon, temp, psal, doxy, variable]}
    flag = flag_missing_extreme * flag_counting_measurement  
    return dict_float, flag


def make_dataset(path_float, variable):
    name_list = pd.read_csv(path_float + '/Float_Index.txt', header=None).to_numpy()[:, 0].tolist()
    datetime_list = pd.read_csv(path_float + '/Float_Index.txt', header=None).to_numpy()[:, 3].tolist()
    dict_ds_accepted = dict()
    dict_ds_removed = dict()
    for i in range(np.size(name_list)):
        path = path_float + "/" + name_list[i]
        if not os.path.exists(path):
            continue
        date_time = datetime_list[i]
        dict_single_float, flag = make_dict_single_float(path, date_time, variable)
        if flag == 1:
            dict_ds_accepted = {**dict_ds_accepted, **dict_single_float}
        if flag == 0:
            dict_ds_removed = {**dict_ds_removed, **dict_single_float}
    return dict_ds_accepted, dict_ds_removed


def make_pandas_df(path_float, variable):
    dict_ds_accepted, dict_ds_removed = make_dataset(path_float, variable)
    print("ds accepted keys", dict_ds_accepted.keys())
    print("ds removed keys", dict_ds_removed.keys())
    dict_ds_accepted = shuffle_dict(dict_ds_accepted)
    for dict_rem in dict_ds_removed:
        pd_ds_removed = pd.DataFrame(dict_rem, index=['year', 'day_rad', 'lat', 'lon', 'temp', 'psal', 'doxy', variable])
        pd_ds_removed.to_csv(f'../{variable}/{dict_rem.keys()[0]}/float_ds_sf_removed.csv')
    train_frac = 0.8
    train_size = int(train_frac * len(dict_ds_accepted))
    val_size = len(dict_ds_accepted) - train_size
    dict_ds_accepted_train = dict_ds_accepted.copy()
    dict_ds_accepted_test = dict(list(dict_ds_accepted.items())[:val_size])
    for key_test in dict_ds_accepted_test.keys():
        dict_ds_accepted_train.pop(key_test, None)
    for dict_train in dict_ds_accepted_train:
        pd_ds_accepted_train = pd.DataFrame(dict_train, index=['year', 'day_rad', 'lat', 'lon', 'temp', 'psal', 'doxy', variable])
        pd_ds_accepted_train.to_csv(f'../{variable}/{dict_train.keys()[0]}/float_ds_sf_train.csv')
    for dict_test in dict_ds_accepted_test:
        pd_ds_accepted_test = pd.DataFrame(dict_test, index=['year', 'day_rad', 'lat', 'lon', 'temp', 'psal', 'doxy', variable])
        pd_ds_accepted_test.to_csv(f'../{variable}/{dict_test.keys()[0]}/float_ds_sf_test.csv')
    return


def make_toy_dataset(path_float, variable):
    name_list = pd.read_csv(path_float + '/Float_Index.txt', header=None).to_numpy()[:, 0].tolist()
    datetime_list = pd.read_csv(path_float + '/Float_Index.txt', header=None).to_numpy()[:, 3].tolist()
    dict_ds = dict()
    for i in range(int(np.size(name_list) / 50)):
        path = path_float + "/" + name_list[i]
        if not os.path.exists(path):
            continue
        date_time = datetime_list[i]
        dict_single_float, flag = make_dict_single_float(path, date_time, variable)
        dict_ds = {**dict_ds, **dict_single_float}
    return dict_ds


def make_pandas_toy_df(path_float, variable):
    dict_ds = make_toy_dataset(path_float, variable)
    for ds in dict_ds:
        pd_ds = pd.DataFrame(ds, index=['year', 'day_rad', 'lat', 'lon', 'temp', 'psal', 'doxy', variable])
        pd_ds.to_csv(f'../{variable}/{ds.keys()[0]}/toy_ds_sf.csv')
    return


def make_data_n_tuple(tensor_low_res_str, i):
    clean_tensor_str = tensor_low_res_str.replace("tensor([", "").replace("])", "")
    tensor_values = [float(x.strip()) for x in clean_tensor_str.split(",")]
    tensor = torch.tensor(tensor_values)
    triplets_list = [tensor.unfold(0, 7, 3)[i].clone() for i in range(2)]
    return triplets_list[i]


def make_pandas_n_tuple_df(path, name_csv_file, variable, flag_complete=1):
    df = pd.read_csv(os.path.join(path, name_csv_file) + ".csv", index_col=0)
    transformations = [lambda x: make_data_n_tuple(x, 0), 
                       lambda x: make_data_n_tuple(x, 1)]  #,
                       #lambda x: generate_data_triplets(x, 2)]    
    transformed_cols = []
    for func in transformations:
        df_trans = df.copy()
        df_trans.loc[variable] = df.loc[variable].apply(func)
        transformed_cols.append(df_trans)
    df_final = pd.concat(transformed_cols, axis=1)
    df_final.to_csv(f'../{variable}/{name_csv_file[:-8]}_triplets.csv')
    return None