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

def list_day_year_df(dataframe):
    pass

def from_day_rad_to_day(day_rad):
    day = (day_rad * 365) / (2 * np.pi)
    return day


def triplets_generation_check(triplets_dataset_file):
    df = pd.read_csv(triplets_dataset_file, header=0)
    search_string = "SD6903823_002"
    matching_cols = [col for col in df.columns if search_string in col]
    print(f"Columns containing '{search_string}':")
    print(matching_cols)
    print("\nData from matching columns:")
    for i in range(len(matching_cols)):
        print(df[matching_cols].iloc[2, i])
    


def make_ds(variable, input_float_directory, flag_complete=1, flag_toy=0):
    from make_float_ds import make_pandas_df, make_pandas_toy_df

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


def make_ds_low_res(variable, path_input_csv, name_input_csv, flag_complete=1, flag_toy=0):
    from make_float_ds import make_dataset_lower_resolution

    if not os.path.exists(f"../{variable}/"):
        os.mkdir(f"../{variable}/")

    if flag_complete and not os.path.exists(f"../{variable}/{name_input_csv}_20_low_res.csv"):
        print("hello making complete ds low res...")
        make_dataset_lower_resolution(path_input_csv, name_input_csv, variable, flag_complete)
        print("superfloat complete ds low res created")

    if flag_toy and not os.path.exists(f"../{variable}/{name_input_csv}_20_low_res.csv"):
        print("hello making toy ds low res...")
        make_dataset_lower_resolution(path_input_csv, name_input_csv, variable, flag_complete)
        print("superfloat toy ds low res created")
    return


def make_ds_triplets(variable, path_input_csv, name_input_csv, flag_complete=1, flag_toy=0):
    from make_float_ds import make_dataset_triplets

    if not os.path.exists(f"../{variable}/"):
        os.mkdir(f"../{variable}/")

    if flag_complete and not os.path.exists(f"../{variable}/{name_input_csv[:-8]}_triplets.csv"):
        print("hello making complete ds...")
        make_dataset_triplets(path_input_csv, name_input_csv, variable, flag_complete)
        print("superfloat complete ds created")

    if flag_toy and not os.path.exists(f'../{variable}/{name_input_csv[:-8]}_triplets.csv'):
        print("hello making toy ds triplets...")
        make_dataset_triplets(path_input_csv, name_input_csv, variable, flag_complete)
        print("superfloat toy ds triplets created")
    return
