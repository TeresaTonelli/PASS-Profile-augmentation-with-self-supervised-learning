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


dict_physics = {"temp" : "TEMP", "psal": "PSAL", "doxy":"DOXY"}


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


def make_dict_single_phys_float(path, date_time, var_phys):
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
    if dict_physics[var_phys] not in ds.variables.keys():
        return dict(), 0
    physic_var = dict_physics[var_phys]
    print("physic var", physic_var)
    phys_var_df = ds[f"{physic_var}"][:].data[:]
    pres_phys_var_df = ds[f"PRES_{physic_var}"][:].data[:]
    if physic_var == "TEMP":    #QUESTO è da chiedere a GP, io ho solo copiato quelli del nitrato
        flag_missing_extreme = 1
        flag_counting_measurement = 1
    max_pres = dict_max_pressure[physic_var]
    interval = dict_interval[physic_var]
    var = discretize(pres_phys_var_df, phys_var_df, max_pres, interval)
    name_float = path.split("/")[-1][:-3]
    if var is None:
        return dict(), 0
    dict_float = {name_float: [year, day_rad, lat, lon, var]}
    flag = flag_missing_extreme * flag_counting_measurement  
    return dict_float, flag


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
    print("name float", name_float)
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
    dict_ds_accepted = shuffle_dict(dict_ds_accepted)
    pd_ds_removed = pd.DataFrame(dict_ds_removed, index=['year', 'day_rad', 'lat', 'lon', 'temp', 'psal', 'doxy', variable])
    pd_ds_accepted = pd.DataFrame(dict_ds_accepted, index=['year', 'day_rad', 'lat', 'lon', 'temp', 'psal', 'doxy', variable])
    pd_ds_accepted.to_csv(f'../{variable}/float_ds_sf_accepted.csv')
    pd_ds_removed.to_csv(f'../{variable}/float_ds_sf_removed.csv')
    return


def make_toy_dataset(path_float, variable):
    name_list = pd.read_csv(path_float + '/Float_Index.txt', header=None).to_numpy()[:, 0].tolist()
    datetime_list = pd.read_csv(path_float + '/Float_Index.txt', header=None).to_numpy()[:, 3].tolist()
    dict_ds = dict()
    for i in range(int(np.size(name_list) / 5)):
        path = path_float + "/" + name_list[i]
        if not os.path.exists(path):
            continue
        date_time = datetime_list[i]
        dict_single_float, flag = make_dict_single_float(path, date_time, variable)
        dict_ds = {**dict_ds, **dict_single_float}
    return dict_ds


def make_pandas_toy_df(path_float, variable):
    dict_ds = make_toy_dataset(path_float, variable)
    pd_ds = pd.DataFrame(dict_ds, index=['year', 'day_rad', 'lat', 'lon', 'temp', 'psal', 'doxy', variable])
    pd_ds.to_csv(f'../{variable}/toy_ds_sf.csv')
    return


def make_per_float_df(full_floats_ds, variable):
    df = pd.read_csv(full_floats_ds, header=0)
    list_full_floats = [name for name in df.columns if name != '']
    list_devices_float = [name.split("_")[0] for name in list_full_floats]
    list_devices_float = list(dict.fromkeys(list_devices_float))
    list_devices_float.remove('Unnamed: 0')
    if df.columns[0] == '':
        df = df.drop(df.columns[0], axis=1)
    for float_device in list_devices_float:
        filtered_df = df.loc[:, df.columns.str.startswith(float_device)]
        filtered_df.insert(0, 'Unnamed: 0', ['year', 'day_rad', 'lat', 'lon', 'temp', 'psal', 'doxy', variable])
        os.makedirs(f'../{variable}/{float_device}', exist_ok=True)
        filtered_df.to_csv(f'../{variable}/{float_device}/float_ds_df_accepted.csv', index=False)
    return None 


def make_per_float_toy_df(full_floats_ds, variable):
    df = pd.read_csv(full_floats_ds, header=0)
    list_full_floats = [name for name in df.columns if name != '']
    print("list full floats", list_full_floats)
    list_devices_float = [name.split("_")[0] for name in list_full_floats]
    list_devices_float = list(dict.fromkeys(list_devices_float))
    list_devices_float.remove('Unnamed: 0')
    print("list devices floats", list_devices_float)
    if df.columns[0] == '':
        df = df.drop(df.columns[0], axis=1)
    for float_device in list_devices_float:
        print("float device number", float_device)
        filtered_df = df.loc[:, df.columns.str.startswith(float_device)]
        #filtered_df.insert(0, 'Unnamed: 0', ['year', 'day_rad', 'lat', 'lon', 'temp', 'psal', 'doxy', 'CHLA'])
        filtered_df.insert(0, 'Unnamed: 0', ['year', 'day_rad', 'lat', 'lon', 'temp', 'psal', 'doxy', variable])
        os.makedirs(f'../{variable}/{float_device}', exist_ok=True)
        filtered_df.to_csv(f'../{variable}/{float_device}/toy_ds_df.csv', index=False)
    return None


def from_day_rad_to_day(day_rad):
    day = (day_rad * 365) / (2 * np.pi)
    return day

def list_year_day_df(sorted_dataframe_csv):
    df_sorted = pd.read_csv(sorted_dataframe_csv, index_col=0)   #NON SO SE DEVO USARE INDEX_COL = 0, DA CONTROLLARE
    df_sorted = df_sorted.T 
    year_day_rad_list = list(df_sorted[["year", "day_rad"]].itertuples(index=False, name=None))
    year_day_list = [from_day_rad_to_day(float(year_day_rad_list[i][1])) for i in range(len(year_day_rad_list))]
    return year_day_rad_list, year_day_list 

def make_sorted_year_day_df(dataframe_variable_dir, variable, flag_toy=1):
    for subfolder in os.listdir(dataframe_variable_dir):
        if subfolder == 'Unnamed: 0':
            continue
        subfolder_path = os.path.join(dataframe_variable_dir, subfolder)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                mode = filename.split("_")[-1].split(".")[0]
                df = pd.read_csv(os.path.join(subfolder_path, filename), index_col=0)    #NON SO SE DEVO USARE INDEX_COL = 0, DA CONTROLLARE 
                df = df.T
                df_sorted = df.sort_values(by=['year', 'day_rad'])
                df_sorted = df_sorted.T
                if flag_toy == 0:
                    df_sorted.to_csv(f'/{subfolder_path}/float_ds_sf_{mode}_ts.csv')
                elif flag_toy == 1:
                    df_sorted.to_csv(f'/{subfolder_path}/toy_ds_df_time_sorted.csv')
    return None


def make_data_n_tuple(dataframe_row, type, i):
    if type == "scalar":
        full_scalar = []
        for cell in dataframe_row:
            value = float(cell)
            full_scalar.append(value)
        output_n_tuples = [full_scalar[j:j+7] for j in range(0, dataframe_row.shape[0] -7 +1)]   
    if type == "tensor":
        tensors = []
        for cell in dataframe_row:
            clean = cell.replace("tensor([", "").replace("])", "")
            values = [float(x.strip()) for x in clean.split(",")]
            tensor = torch.tensor(values).unsqueeze(1)  
            tensors.append(tensor)
        full_tensor = torch.cat(tensors, dim=1)
        output_n_tuples = [full_tensor[:, j:j+7] for j in range(0, full_tensor.shape[1] - 7 + 1)]
    return output_n_tuples[i] 


def is_arithmetic_sequence(tup):   
    diffs = [tup[i+1] - tup[i] for i in range(len(tup) - 1)]
    print("days int", [int(from_day_rad_to_day(tup[i])) for i in range(len(tup) - 1)])
    print("diffs", diffs)
    print("condition diffs", [np.isclose(d, diffs[0], rtol=2 * 1e-1, atol=1e-2) for d in diffs])
    print("condition diffs", all(np.isclose(d, diffs[0], rtol=2 * 1e-1, atol=1e-2) for d in diffs))
    return all(np.isclose(d, diffs[0], rtol=2 * 1e-1, atol=1e-2) for d in diffs)


def make_pandas_n_tuple_df(path, name_csv_file, variable, list_features, flag_complete=1):
    df = pd.read_csv(os.path.join(path, name_csv_file) + ".csv", index_col=0)
    list_all_variables = []
    feature_names = []
    for feature in list_features + [variable]:
        row = df.loc[feature]
        list_feature = []
        if feature in ["year", "day_rad", "lat", "lon"]:
            feature_type = "scalar"
        elif feature in ["temp", "psal", "doxy", "CHLA", "NITRATE", "BBP700"]:
            feature_type = "tensor"
        m = df.shape[1] -7 +1
        for i in range(m): 
            new_tensor = make_data_n_tuple(row, feature_type, i)
            if feature_type == "tensor": 
                list_feature.append(new_tensor.flatten().tolist())  
            elif feature_type == "scalar":
                list_feature.append(new_tensor)
        feature_names.append(f"{feature}")
        list_all_variables.append(list_feature)
    # Create final DataFrame
    df_final = pd.DataFrame(list_all_variables, index=feature_names)
    df_final.columns = [f"{df.columns[0].split("_")[0]}_{j}" for j in range(df_final.shape[1])]  

    feature_day_row = df_final.loc['day_rad']
    arithmetic_mask = feature_day_row.apply(is_arithmetic_sequence)
    df_final_time_step = df_final.loc[:, arithmetic_mask]

    # Save to CSV
    df_final_time_step.to_csv(os.path.join(path, f"{name_csv_file}_nt.csv"))
    print("df_final shape:", df_final_time_step.shape)

    return df_final


def dataframe_merging(list_dataframe_files, path, flag):
    list_dataframe = []
    list_dataframe_shapes = []
    for csv_dataframe in list_dataframe_files:
        if not os.path.exists(csv_dataframe):
            continue
        print("csv dataframe", csv_dataframe)
        df = pd.read_csv(csv_dataframe, index_col=0)
        list_dataframe.append(df)
        list_dataframe_shapes.append(df.shape[1])
    df_merged = pd.concat(list_dataframe, axis=1)
    assert df_merged.shape[1] == sum(list_dataframe_shapes)
    if flag == 1:
        name_csv = "complete"
    elif flag == 0:
        name_csv = "toy"
    print("df merged shape", df_merged.shape)
    df_merged.to_csv(os.path.join(path, f"{name_csv}_merged_n_tuple.csv"))
    return None


def data_merge(list_dataframe_files, list_ext_test_dataframe, path, flag):
    list_dataframe_train = []
    list_dataframe_test = []
    list_dataframe_train_shapes = []
    list_dataframe_test_shapes = []
    for csv_dataframe in list_dataframe_files:
        if not os.path.exists(csv_dataframe):
            continue
        df = pd.read_csv(csv_dataframe, index_col=0) 
        k = int(0.1 * df.shape[1])
        test_indexes = random.sample(range(1, df.shape[1]), k)  
        train_indexes = [j for j in range(df.shape[1]) if j not in test_indexes]
        df_train = df.iloc[:, train_indexes].copy()
        df_test  = df.iloc[:, test_indexes].copy()
        assert df.shape[1] == df_train.shape[1] + df_test.shape[1]
        list_dataframe_train.append(df_train)
        list_dataframe_test.append(df_test)
        list_dataframe_train_shapes.append(df_train.shape[1])
        list_dataframe_test_shapes.append(df_test.shape[1])
    df_train_merged = pd.concat(list_dataframe_train, axis=1)
    assert df_train_merged.shape[1] == sum(list_dataframe_train_shapes)
    for csv_ext_dataframe in list_ext_test_dataframe:
        if not os.path.exists(csv_ext_dataframe):
            continue
        df_ext = pd.read_csv(csv_ext_dataframe, index_col=0)
        list_dataframe_test.append(df_ext)
        list_dataframe_test_shapes.append(df_ext.shape[1]) 
    df_test_merged = pd.concat(list_dataframe_test, axis=1)
    df_test_merged.insert(0, "Unnamed_0", df_ext.iloc[:, 0])
    assert df_test_merged.shape[1] == sum(list_dataframe_test_shapes) + 1
    if flag == 1:
        name_csv = "complete"
    elif flag == 0:
        name_csv = "toy"
    print("df train merged shape", df_train_merged.shape)
    print("df test merged shape", df_test_merged.shape)
    df_train_merged.to_csv(os.path.join(path, f"{name_csv}_train_mnt.csv"))
    df_test_merged.to_csv(os.path.join(path, f"{name_csv}_test_mnt.csv"))
    return None


def split_time_step_dataframe(full_dataframe_file, variable):
    df = pd.read_csv(full_dataframe_file, index_col=0)
    time_steps_columns_indexes = {}
    time_steps_dfs = {}
    day_row = df.iloc[1]

    for col_index, col_name in enumerate(df.columns):
        day_tuple = day_row[col_name]  # This should be a tuple of 7 elements
        day_tuple = day_tuple.replace("[", "").replace("]", "")
        day_tuple = [float(x.strip()) for x in day_tuple.split(",")]

        # Compute the time step (assuming constant spacing)
        time_step = round(from_day_rad_to_day(day_tuple[1]) - from_day_rad_to_day(day_tuple[0]))

        # Initialize storage if new time step encountered
        if time_step not in time_steps_columns_indexes:
            time_steps_columns_indexes[time_step] = []

        # Store column index for this time step
        time_steps_columns_indexes[time_step].append(col_index)

    # Now create the grouped dataframes
    for time_step, col_indexes in time_steps_columns_indexes.items():
        col_names = df.columns[col_indexes]
        time_steps_dfs[time_step] = df[col_names]

    for k in time_steps_dfs.keys():
        print("key", k)
        df = time_steps_dfs[k]
        df.to_csv(os.path.join(f"../{variable}/{k}_time_steps_n_tuple.csv"))

    print("len time steps dfs", len(time_steps_dfs))
    return time_steps_columns_indexes, time_steps_dfs



