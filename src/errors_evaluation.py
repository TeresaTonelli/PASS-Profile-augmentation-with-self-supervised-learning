import numpy as np
import torch
import os
import pandas as pd
from itertools import islice

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
        print("rmse sample shape", rmse_sample.shape)
        print("rmse sample", rmse_sample)
        RMSEs = torch.cat((RMSEs, rmse_sample), 0)

        with open(dir_RMSE + "/RMSE_file.txt", 'a') as f: 
            f.write(str([rmse_sample[:, i].item() for i in range(rmse_sample.shape[1])]) + '\n')
        

    print("RMSEs final shape", RMSEs.shape)
    assert RMSEs.shape[0] == m+1
    RMSEs = RMSEs[1:, :]

    with open(dir_RMSE + "/RMSE_file.txt", 'a') as f:
        f.write(f"mean: {[torch.mean(RMSEs[:, i]) for i in range(rmse_sample.shape[1])]} \n") 
        f.write(f"median: {[torch.median(RMSEs[:, i]) for i in range(rmse_sample.shape[1])]} \n")

    return None