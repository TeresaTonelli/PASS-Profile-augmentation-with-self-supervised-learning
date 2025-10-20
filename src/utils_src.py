import os
import pandas as pd


def create_directories_f(list_structure, my_date):
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


def save_ds_info_f(variable, flag_toy, batch_size, epochs, lr, save_dir, alpha_smooth_reg):
    dict_info = {'variable': [variable], 
                 'is_toy': [flag_toy],
                 'batch_size': [batch_size],
                 'epoch': [epochs],
                 'lr': [lr],
                 'alpha_smooth_reg': alpha_smooth_reg}
    pd_ds = pd.DataFrame(dict_info)
    pd_ds.to_csv(save_dir + '/info.csv')
    return