import numpy as np
import torch
import os
import pandas as pd
from itertools import islice
import matplotlib.pyplot as plt

from NN_architecture import UNet1D
from utils_plots import from_loader_data_to_train_data_f


def RMSE_comparison(ds, dir, variable, ep=100, mov_average=False, m=50):
    dir_model = dir + "/model"
    dir_RMSE = dir + "/error_evaluation"
    if not os.path.exists(dir_RMSE):
        os.makedirs(dir_RMSE)
    dir_info = dir + "/info.csv"
    info_df = pd.read_csv(dir_info)     
    path_model_conv = dir_model + "/model_" + str(ep) + ".pt"
    model = UNet1D()    #.to(device)   #AGGIUNTO ORA QUESTO
    model.load_state_dict(torch.load(path_model_conv))
    model.eval()

    RMSEs = torch.zeros([1, 5])

    for sample in list(islice(ds, m)):
        var = sample[1]
        testing_x_tmp = from_loader_data_to_train_data_f(sample)
        testing_x_sx, testing_x_dx = testing_x_tmp  
        out_d = model(testing_x_sx[:, [0,3], :])  
        out_1pd = model(testing_x_dx[:, [0,3], :]) 
        out_2d = model(torch.cat((out_d[:, 0:1, :], out_1pd[:, 0:1, :]), 1))
        out_1p2d = model(torch.cat((out_d[:, 1:2, :], out_1pd[:, 1:2, :]), 1))
        test_profile_d = out_d[:,0:1,:].squeeze()
        test_profile_1_2d = out_1pd[:,1:2,:].squeeze()
        test_profile_2d = ((out_d[:,1:2,:] + out_2d[:,0:1,:])/2).squeeze()
        test_profile_1_d = ((out_1pd[:,0:1,:] + out_1p2d[:,1:2,:])/2).squeeze()
        test_profile_1 = ((out_2d[:,1:2,:]+out_1p2d[:,0:1,:])/2).squeeze()

        rmse_sample = torch.unsqueeze(torch.tensor([torch.sqrt(torch.nn.functional.mse_loss(var[:, 1:2, :], test_profile_d)),  torch.sqrt(torch.nn.functional.mse_loss(var[:, 2:3, :], test_profile_2d)),
                                    torch.sqrt(torch.nn.functional.mse_loss(var[:, 3:4, :], test_profile_1)), torch.sqrt(torch.nn.functional.mse_loss(var[:, 4:5, :], test_profile_1_d)), 
                                    torch.sqrt(torch.nn.functional.mse_loss(var[:, 5:6, :], test_profile_1_2d))]), 0)
        RMSEs = torch.cat((RMSEs, rmse_sample), 0)

        with open(dir_RMSE + "/RMSE_file.txt", 'a') as f: 
            f.write(str([rmse_sample[:, i].item() for i in range(rmse_sample.shape[1])]) + '\n')
        

    assert RMSEs.shape[0] == m+1
    RMSEs = RMSEs[1:, :]

    with open(dir_RMSE + "/RMSE_file.txt", 'a') as f:
        f.write(f"mean: {[torch.mean(RMSEs[:, i]) for i in range(rmse_sample.shape[1])]} \n") 
        f.write(f"median: {[torch.median(RMSEs[:, i]) for i in range(rmse_sample.shape[1])]} \n")

    return None


def RMSE_float_linear(ds, dir, variable, mov_average=False, m=50): 
    dir_model = dir + "/model"
    dir_RMSE = dir + "/error_evaluation"
    if not os.path.exists(dir_RMSE):
        os.makedirs(dir_RMSE)
    dir_info = dir + "/info.csv"
    info_df = pd.read_csv(dir_info) 

    RMSEs = torch.zeros([1, 4])

    for sample in list(islice(ds, m)):
        var = sample[1]
        testing_x_tmp = from_loader_data_to_train_data_f(sample)
        testing_x_sx, testing_x_dx = testing_x_tmp  

        out_1_linear = (2/3)*testing_x_sx[:, 0:1, :] + (1/3)*testing_x_sx[:, 3:, :]
        out_2_linear = (1/3)*testing_x_sx[:, 0:1, :] + (2/3)*testing_x_sx[:, 3:, :]
        out_4_linear = (2/3)*testing_x_dx[:, 0:1, :] + (1/3)*testing_x_dx[:, 3:, :]
        out_5_linear = (1/3)*testing_x_dx[:, 0:1, :] + (2/3)*testing_x_dx[:, 3:, :]

        rmse_sample = torch.unsqueeze(torch.tensor([torch.sqrt(torch.nn.functional.mse_loss(var[:, 1:2, :], out_1_linear)),  torch.sqrt(torch.nn.functional.mse_loss(var[:, 2:3, :], out_2_linear)),
                                    torch.sqrt(torch.nn.functional.mse_loss(var[:, 4:5, :], out_4_linear)), torch.sqrt(torch.nn.functional.mse_loss(var[:, 5:6, :], out_5_linear))]), 0)
        RMSEs = torch.cat((RMSEs, rmse_sample), 0)

        with open(dir_RMSE + "/RMSE_float_linear_file.txt", 'a') as f: 
            f.write(str([rmse_sample[:, i].item() for i in range(rmse_sample.shape[1])]) + '\n')

    assert RMSEs.shape[0] == m+1
    RMSEs = RMSEs[1:, :]

    with open(dir_RMSE + "/RMSE_float_linear_file.txt", 'a') as f:
        f.write(f"mean: {[torch.mean(RMSEs[:, i]) for i in range(rmse_sample.shape[1])]} \n") 
        f.write(f"median: {[torch.median(RMSEs[:, i]) for i in range(rmse_sample.shape[1])]} \n")

    return None
    

def error_evaluation_boxplots(RMSE_txt, RMSE_linear_txt, save_dir):
    with open(RMSE_txt, 'r') as f:
        list_rmse_samples = f.readlines()
        list_rmse_samples = [[float(x.strip()) for x in rmse_sample.strip().strip("[]").split(",")]for rmse_sample in list_rmse_samples[:-2]]
    rmse_samples_1 = [rmse_sample[0] for rmse_sample in list_rmse_samples]
    rmse_samples_2 = [rmse_sample[1] for rmse_sample in list_rmse_samples]
    rmse_samples_4 = [rmse_sample[3] for rmse_sample in list_rmse_samples]
    rmse_samples_5 = [rmse_sample[4] for rmse_sample in list_rmse_samples]
    with open(RMSE_linear_txt, 'r') as f:
        list_rmse_linear_samples = f.readlines()
        list_rmse_linear_samples = [[float(x.strip()) for x in rmse_sample.strip().strip("[]").split(",")]for rmse_sample in list_rmse_linear_samples[:-2]]
        assert len(list_rmse_linear_samples) == 50
    rmse_linear_samples_1 = [rmse_linear_sample[0] for rmse_linear_sample in list_rmse_linear_samples]
    rmse_linear_samples_2 = [rmse_linear_sample[1] for rmse_linear_sample in list_rmse_linear_samples]
    rmse_linear_samples_4 = [rmse_linear_sample[2] for rmse_linear_sample in list_rmse_linear_samples]
    rmse_linear_samples_5 = [rmse_linear_sample[3] for rmse_linear_sample in list_rmse_linear_samples]

    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    bp_1 = axes[0, 0].boxplot([rmse_samples_1, rmse_linear_samples_1], labels=["1DUnet", "Linear Interpolation"], patch_artist=True)
    axes[0, 0].set_title("Time 1")
    bp_1['boxes'][0].set_facecolor('skyblue')
    bp_1['boxes'][1].set_facecolor('peachpuff')
    bp_2 = axes[0, 1].boxplot([rmse_samples_2, rmse_linear_samples_2], labels=["1DUnet", "Linear Interpolation"], patch_artist=True)
    axes[0, 1].set_title("Time 2")
    bp_2['boxes'][0].set_facecolor('skyblue')
    bp_2['boxes'][1].set_facecolor('peachpuff')
    bp_4 = axes[1, 0].boxplot([rmse_samples_4, rmse_linear_samples_4], labels=["1DUnet", "Linear Interpolation"], patch_artist=True)
    axes[1, 0].set_title("Time 4")
    bp_4['boxes'][0].set_facecolor('skyblue')
    bp_4['boxes'][1].set_facecolor('peachpuff')
    bp_5 = axes[1, 1].boxplot([rmse_samples_5, rmse_linear_samples_5], labels=["1DUnet", "Linear Interpolation"], patch_artist=True)
    axes[1, 1].set_title("Time 5")
    bp_5['boxes'][0].set_facecolor('skyblue')
    bp_5['boxes'][1].set_facecolor('peachpuff')

    plt.tight_layout()
    plt.savefig(save_dir + "/error_comparison_boxplots.png")
    plt.close()
