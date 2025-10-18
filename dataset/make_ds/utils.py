import os
import random
import numpy as np
import pandas as pd
import torch
import json


def create_directories(list_structure, my_date):
    save_dir = os.getcwd()
    for folder in list_structure:
        save_dir = save_dir + "/" + str(folder) 
        os.makedirs(save_dir, exist_ok=True)
    folders_name = os.listdir(save_dir)
    date_save_dir = save_dir + "/" + str(my_date) 
    if not os.path.exists(date_save_dir):
        os.makedirs(date_save_dir, exist_ok=True)
    else:
        m = sum(str(my_date) in item for item in folders_name) 
        date_save_dir = save_dir + "/" + str(my_date) + "_" + str(m)
        os.makedirs(date_save_dir, exist_ok=True)
    return date_save_dir


def shuffle_dict(my_dict):
    items = list(my_dict.items())  
    random.shuffle(items)
    return dict(items)


def triplets_generation_check(triplets_dataset_file):
    df = pd.read_csv(triplets_dataset_file, header=0)
    search_string = "SD6903823_002"
    matching_cols = [col for col in df.columns if search_string in col]
    print(f"Columns containing '{search_string}':")
    print(matching_cols)
    print("\nData from matching columns:")
    for i in range(len(matching_cols)):
        print(df[matching_cols].iloc[2, i])
    

def n_tuple_tensors_generation_check(original_dataset_file, time_sorted_dataset_file, n_tuples_dataset_file):
    df_original = pd.read_csv(original_dataset_file, header=0)
    df_time_sorted = pd.read_csv(time_sorted_dataset_file, header=0)
    df_n_tuples = pd.read_csv(n_tuples_dataset_file, header=0)
    variables = df_time_sorted.index.tolist()
    float_n_tuples_latitude = [float(x.strip()) for x in df_n_tuples.iloc[2, 1].strip("[]").split(",")]
    #print("float_n_tuples latitudes", float_n_tuples_latitude)
    #print("n tuples CHLA", df_n_tuples.iloc[7, 1])
    chla_list = [float(x.strip()) for x in df_n_tuples.iloc[7, 1].strip("[]").split(",")]
    chla_tensor = torch.tensor(chla_list)
    #print("chla tensor", chla_tensor)
    #print("type CHLA", type(chla_tensor))
    #print("shape CHLA", chla_tensor.shape)
    for i in range(1, 8):
        #print("time sorted latitude", df_time_sorted.iloc[variables[2], i])
        print("latitude euqality", float_n_tuples_latitude[i-1] == float(df_time_sorted.iloc[variables[2], i]))
        clean = df_time_sorted.iloc[variables[7], i].replace("tensor([", "").replace("])", "")
        values = [float(x.strip()) for x in clean.split(",")]
        tensor = torch.tensor(values)
        #print("tensor CHLA", tensor)
        #print("tensor CHLA type", type(tensor))
        #print("shape CHLA tensor", tensor.shape)
        #print("type single element tensor", type(chla_tensor[1]))
        #print("tensor CHLA 1", [chla_tensor[j].item() for j in range(1400) if j%7 == i-1])
        #print("tensor CHLA 2", tensor.tolist())
        print("CHLA check equality", [chla_tensor[j].item() for j in range(1400) if j%7 == i-1] == tensor.tolist())




def make_ds(variable, input_float_directory, flag_complete=1, flag_toy=0):
    from make_ds_float import make_pandas_df, make_pandas_toy_df

    if not os.path.exists(f"../{variable}/"):
        os.mkdir(f"../{variable}/")

    if flag_complete and not os.path.exists(f"../{variable}/float_ds_sf_train.csv"):
        print("hello making complete ds...")
        make_pandas_df(input_float_directory, variable=variable)
        print("superfloat complete ds created")

    if flag_toy and not os.path.exists(f"../{variable}/toy_ds_sf.csv"):
        print("hello making toy ds...")
        make_pandas_toy_df(input_float_directory, variable=variable)
        print("superfloat toy ds created")
    return


def make_ds_float_devices(variable, csv_data_directory, flag_complete=1, flag_toy=0):
    from make_ds_float import make_per_float_df, make_per_float_toy_df

    if not os.path.exists(f"../{variable}/"):
        os.mkdir(f"../{variable}/")

    if flag_complete and not os.path.exists(f"../{variable}/SD6903823/float_ds_df_train.csv"):
        print("hello making complete ds...")
        make_per_float_df(csv_data_directory, variable)
        print("superfloat complete ds float devices created")

    if flag_toy and not os.path.exists(f"../{variable}/SD6903823/toy_ds_df.csv"):
        print("hello making toy ds device float...")
        make_per_float_toy_df(csv_data_directory, variable)
        print("superfloat toy ds float devices created")
    return


def make_ds_time_sort(variable, var_data_directory, sorted_float_csv, flag_complete=1, flag_toy=0):
    from make_ds_float import make_sorted_year_day_df, list_year_day_df

    if not os.path.exists(f"../{variable}/"):
        os.mkdir(f"../{variable}/")

    if flag_complete and not os.path.exists(f"../{variable}/SD6903823/float_ds_df_train_time_sorted.csv"):
        print("hello making complete time order ds...")
        make_sorted_year_day_df(var_data_directory, variable)
        print(list_year_day_df(sorted_float_csv)[0])
        print(list_year_day_df(sorted_float_csv)[1])
        print("superfloat complete ds time order created")

    if flag_toy and not os.path.exists(f"../{variable}/SD6903823/toy_ds_df_time_sorted.csv"):
        print("hello making toy ds time order float...")
        make_sorted_year_day_df(var_data_directory, variable)
        print(list_year_day_df(sorted_float_csv)[0])
        print(list_year_day_df(sorted_float_csv)[1])
        print("superfloat toy ds time order created")


def make_ds_n_tuples(variable, list_features, path_input_csv, name_input_csv, flag_complete=1, flag_toy=0):
    from make_ds_float import make_pandas_n_tuple_df

    if not os.path.exists(f"../{variable}/"):
        os.mkdir(f"../{variable}/")

    if flag_complete and not os.path.exists(f"../{variable}/SD6903823/{name_input_csv}_n_tuple.csv"):
        print("hello making complete n_tuples ds...")
        make_pandas_n_tuple_df(path_input_csv, name_input_csv, variable, list_features, flag_complete)
        print("superfloat complete ds n_tuples created")

    if flag_toy and not os.path.exists(f'../{variable}/SD6903823/{name_input_csv}_n_tuple.csv'):
        print("hello making toy ds n_tuples...")
        make_pandas_n_tuple_df(path_input_csv, name_input_csv, variable, list_features, flag_complete)
        print("superfloat toy ds n_tuples created")
    return


def make_ds_merging(variable, list_devices, path_input_csv, name_input_csv, flag_complete=1, flag_toy=0):
    from make_ds_float import dataframe_merging

    if not os.path.exists(f"../{variable}/"):
        os.mkdir(f"../{variable}/")

    if flag_complete and not os.path.exists(f"../{variable}/{name_input_csv}_merged_n_tuple.csv"):
        print("hello making complete n_tuples merging ds...")
        list_dataframe_files = []
        dataframe_merging(list_dataframe_files, path_input_csv, flag_complete)
        print("superfloat complete ds n_tuples merging created")

    if flag_toy and not os.path.exists(f'../{variable}/{name_input_csv}_merged_n_tuple.csv'):
        print("hello making toy ds n_tuples merging...")
        list_dataframe_files = [os.path.abspath(os.path.join("..", variable,  list_devices[j], "toy_ds_df_time_sorted_n_tuple.csv")) for j in range(len(list_devices))]
        dataframe_merging(list_dataframe_files, path_input_csv, flag_complete)
        print("superfloat toy ds n_tuples merging created")
    return