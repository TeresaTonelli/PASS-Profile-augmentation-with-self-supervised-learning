import os
import pandas as pd
import torch 


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


def create_directories_f_1(list_structure, my_date):
    save_dir = "/leonardo_work/OGS23_PRACE_IT_1/ttonelli/SSint_results"
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


def EMD_loss(p,q):
    """
    Earth Mover's Distance loss function
    """
    p_sum = torch.sum(p, dim=-1, keepdim=True)
    q_sum = torch.sum(q, dim=-1, keepdim=True)
    p = p/ (p_sum  + 1e-8)
    q = q/ (q_sum  + 1e-8)
    CDF_p = torch.cumsum(p, dim=-1)
    CDF_q = torch.cumsum(q, dim=-1)
    emd = torch.sum(torch.abs(CDF_p - CDF_q), dim=-1) * 1 
    return emd.mean()