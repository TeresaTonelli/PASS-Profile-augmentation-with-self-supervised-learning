import os
import random
from datetime import date
import argparse
import torch
import pandas as pd

# Setting the computation device
from utils import make_ds, make_ds_float_devices, make_ds_time_sort, make_ds_n_tuples, triplets_generation_check

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"We will use {device}")

random.seed(123)

# ===== Create the parser
parser = argparse.ArgumentParser()
parser.add_argument('--training_folder', type=str, default="SSint_FLOAT", choices=["SSint_FLOAT"])
parser.add_argument('--input_float_directory', type=str, default='/leonardo_scratch/large/userexternal/ttonelli/OGS/SSint_FLOAT/SUPERFLOAT')
parser.add_argument('--flag_toy', type=bool, default=False)
parser.add_argument('--variable', type=str, default="NITRATE", choices=["NITRATE", "CHLA", "BBP700"])

# ===== Parsing arguments
args = parser.parse_args()
training_folder = args.training_folder
input_float_directory = args.input_float_directory
variable = args.variable
flag_toy = args.flag_toy

# ===== Printing information about the run
print(f"The dataset used is {training_folder}\n"
      f"We used a reduced version of the ds? {bool(flag_toy)}\n"
      f"the input float data directory is {input_float_directory}"
      f"The variable predicted is {variable}")

# ===== Creating the correct dataframe according to the training folder (if it does not exists yet!)
#df = pd.read_csv(os.path.abspath(os.path.join("..", variable, "toy_ds_sf_20_triplets.csv")))
#print("df shape", df.shape)
#part1 = df.iloc[:, :400]  # First 2 columns
#part2 = df.iloc[:, 400:]  # Last 2 columns
#part1.to_csv(os.path.abspath(os.path.join("..", variable, "toy_ds_sf_20_triplets_train.csv")), index=False)
#part2.to_csv(os.path.abspath(os.path.join("..", variable, "toy_ds_sf_20_triplets_test.csv")), index=False)


make_ds(variable=variable, input_float_directory=input_float_directory, flag_complete=abs(1-flag_toy), flag_toy=flag_toy)
path_input_csv = os.path.abspath(os.path.join("..", variable))
name_input_csv = "toy_ds_sf"
name_input_time_sort = "toy_ds_df_time_sorted"
csv_data_directory = os.path.abspath(os.path.join("..", variable, name_input_csv + ".csv"))
make_ds_float_devices(variable, csv_data_directory , flag_complete=abs(1-flag_toy), flag_toy=flag_toy)
make_ds_time_sort(variable, path_input_csv, os.path.abspath(os.path.join("..", variable, 'SD6903823',name_input_time_sort + ".csv")), 
                  flag_complete=abs(1-flag_toy), flag_toy=flag_toy)

float_devices = ['SD6901765','SD6903823', 'SR6903823']
list_features = ['temp', 'psal', 'doxy']
for device in float_devices:
    path_input_time_sorted = os.path.abspath(os.path.join("..", variable, device))
    make_ds_n_tuples(variable, list_features, path_input_time_sorted, name_input_time_sort, flag_complete=abs(1-flag_toy), flag_toy=flag_toy)
#triplets_dataset_file = os.path.abspath(os.path.join("..", variable, "toy_ds_sf_20_triplets.csv"))
#triplets_generation_check(triplets_dataset_file)