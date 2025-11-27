import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from itertools import islice

from NN_architecture import UNet1D
from utils_plots import from_loader_data_to_train_data_f


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dict_features_idx = {"year":0, "day_rad":1, "lat":2, "lon":3, "temp":4, "psal":5, "doxy":6}

def plot_loss_f(losses, n_epochs, snaperiod, path):
    figure(figsize=(10, 6))
    label = 'losses'
    plt.plot(losses, 'orange')
    plt.plot(losses, 'm.', label=label)
    my_xticks = np.arange(0, n_epochs // snaperiod)
    my_xticks_label = np.arange(snaperiod, n_epochs + snaperiod, snaperiod) 
    plt.xticks(my_xticks, my_xticks_label, fontsize=6)
    plt.xlabel('Number of epochs')
    plt.title('Losses over training')
    plt.legend()
    os.makedirs(path, exist_ok=True)
    plt.savefig(path + "/training_loss.png")
    plt.close()
    return None


def torch_moving_average_f(x, window_size):
    kernel = torch.ones(1, 1, window_size, device=x.device) / window_size
    padding = window_size // 2
    input_padded = F.pad(x, (padding, padding), mode='replicate')
    output_ma = F.conv1d(input_padded, kernel, padding=0)
    return output_ma.squeeze()


def plot_profiles_f(ds, dir, variable, multiple_GPUs = False, ep=100, train=False, mov_average=False):
    dir_model = dir + "/model"
    if train == False:
        dir_profile = dir + "/profile"
        if not os.path.exists(dir_profile):
            os.mkdir(path=dir_profile)
    else:
        dir_profile = dir + "/profile_train"
        if not os.path.exists(dir_profile):
            os.mkdir(path=dir_profile)

    dir_info = dir + "/info.csv"
    info_df = pd.read_csv(dir_info)
    max_pres = 200    #dict_max_pressure[variable]
    interval = 1      #dict_interval[variable]

    # Path of the saved models
    print("epoch", ep)
    path_model_conv = dir_model + "/model_" + str(ep) + ".pt"
    print("path conv mod", path_model_conv)

    model = UNet1D() #.to(device)   #AGGIUNTO ORA QUESTO
    model.load_state_dict(torch.load(path_model_conv))
    model.eval()


    #for year, day_rad, lat, lon, temp, psal, doxy, output_variable in ds:
    for sample in list(islice(ds, 500)):
        var = sample[1]
        print("var shape", var.shape) 
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
        test_profile_1 = ((testing_x_sx[:,-1:,:]+out_2d[:,1:2,:]+out_1p2d[:,0:1,:])/3).squeeze()

        depth_output = np.linspace(0, max_pres-1, len(test_profile_1.detach().numpy()))  #prima era [0,0,:]
        depth_variable = np.linspace(0, max_pres-1, len(test_profile_1.detach().numpy()))    #prima era [0,0,:]

        if variable == "BBP700":
            var = var / 1000
            test_profile_1 = test_profile_1 / 1000

        #test_ma_profile = test_profile_1
        #var_ma = var
        if mov_average == True:
            test_profile_d = torch_moving_average_f(test_profile_d.view(1, 1, -1), 3)
            test_profile_2d = torch_moving_average_f(test_profile_2d.view(1, 1, -1), 3)
            test_profile_1= torch_moving_average_f(test_profile_1.view(1, 1, -1), 3)
            test_profile_1_d= torch_moving_average_f(test_profile_1_d.view(1, 1, -1), 3)
            test_profile_1_2d= torch_moving_average_f(test_profile_1_2d.view(1, 1, -1), 3)
            out_2d_ma= torch_moving_average_f(out_2d[:,1:2,:].squeeze().view(1, 1, -1), 3)
            out_1p2d_ma = torch_moving_average_f(out_1p2d[:,0:1,:].squeeze().view(1, 1, -1), 3)
            #var_ma = torch_moving_average_f(var[:,:,:,-1:,3:4].squeeze().view(1, 1, -1), 3)

        fig, axs = plt.subplots(2, 3, figsize=(10, 8))
        axs[0, 0].plot(test_profile_d.detach().numpy(), depth_output, color='blue', label=f"generated {variable}")
        #axs[0, 0].plot(test_profile_d_ma.detach().numpy(), depth_output, color='green', label=f"generated {variable}")
        axs[0, 0].plot(var[:,1:2,:].squeeze().detach().numpy(), depth_output, color='red', label=f"generated {variable}")
        axs[0, 0].invert_yaxis()
        axs[0, 0].set_title('I1 comparison')
        axs[0, 1].plot(test_profile_2d.detach().numpy(), depth_output, color='blue', label=f"generated {variable} ")
        #axs[0, 1].plot(test_profile_2d_ma.detach().numpy(), depth_output, color='green', label=f"generated {variable} ")
        axs[0, 1].plot(var[:,2:3,:].squeeze().detach().numpy(), depth_output, color='red', label=f"generated {variable}")
        axs[0, 1].invert_yaxis()
        axs[0, 1].set_title('I2 comparison')
        axs[0, 2].plot(test_profile_1.detach().numpy(), depth_output, color='blue', label=f"generated {variable}")
        axs[0, 2].plot(var[:,3:4,:].squeeze().detach().numpy(), depth_output, color='red', label=f"generated {variable}")
        axs[0, 2].invert_yaxis()
        axs[0, 2].set_title('I3 comparison')
        axs[1, 0].plot(test_profile_1_d.detach().numpy(), depth_output, color='blue', label=f"generated {variable}")
        axs[1, 0].plot(var[:,4:5,:].squeeze().detach().numpy(), depth_output, color='red', label=f"generated {variable}")
        axs[1, 0].invert_yaxis()
        axs[1, 0].set_title('I4 comparison')
        axs[1, 1].plot(test_profile_1_2d.detach().numpy(), depth_output, color='blue', label=f"generated {variable}")
        axs[1, 1].plot(var[:,5:6,:].squeeze().detach().numpy(), depth_output, color='red', label=f"generated {variable}")
        axs[1, 1].invert_yaxis()
        axs[1, 1].set_title('I5 comparison')
        axs[1, 2].plot(out_2d[:,1:2,:].squeeze().detach().numpy(), depth_output, color='green', label=f"generated {variable}")
        axs[1, 2].plot(out_1p2d[:,0:1,:].squeeze().detach().numpy(), depth_output, color='orange', label=f"generated {variable}")
        axs[1, 2].plot(var[:,3:4,:].squeeze().detach().numpy(), depth_output, color='red', label=f"generated {variable}")
        axs[1, 2].invert_yaxis()
        axs[1, 2].set_title('I3 sx dx comparison')

        plt.legend()
        plt.savefig(dir_profile + f"/profile_{var[:,0:1, 0:1].item()}.png")
        plt.close()


def plot_NN_intermediate_profiles(ds, dir, variable, ep=100, mov_average=False):
    dir_model = dir + "/model"
    dir_profile = dir + "/NN_intermediate_profile"
    if not os.path.exists(dir_profile):
        os.mkdir(path=dir_profile)
    dir_profiles_features = dir + "/profiles_features"
    if not os.path.exists(dir_profiles_features):
        os.mkdir(path=dir_profiles_features)
    dir_info = dir + "/info.csv"
    info_df = pd.read_csv(dir_info)
    max_pres = 200    #dict_max_pressure[variable]
    interval = 1      #dict_interval[variable]
    print("epoch", ep)
    path_model_conv = dir_model + "/model_" + str(ep) + ".pt"
    print("path conv mod", path_model_conv)
    model = UNet1D() #.to(device)   #AGGIUNTO ORA QUESTO
    model.load_state_dict(torch.load(path_model_conv))
    model.eval()

    for sample in list(islice(ds, 500)):
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
        test_profile_1 = ((testing_x_sx[:,-1:,:]+out_2d[:,1:2,:]+out_1p2d[:,0:1,:])/3).squeeze()

        depth_output = np.linspace(0, max_pres-1, len(test_profile_1.detach().numpy())) 
        depth_variable = np.linspace(0, max_pres-1, len(test_profile_1.detach().numpy()))    

        if variable == "BBP700":
            var = var / 1000
            test_profile_1 = test_profile_1 / 1000

        #test_ma_profile = test_profile_1
        #var_ma = var
        if mov_average == True:
            test_profile_d = torch_moving_average_f(test_profile_d.view(1, 1, -1), 3)
            test_profile_2d = torch_moving_average_f(test_profile_2d.view(1, 1, -1), 3)
            test_profile_1= torch_moving_average_f(test_profile_1.view(1, 1, -1), 3)
            test_profile_1_d= torch_moving_average_f(test_profile_1_d.view(1, 1, -1), 3)
            test_profile_1_2d= torch_moving_average_f(test_profile_1_2d.view(1, 1, -1), 3)
            out_2d_ma= torch_moving_average_f(out_2d[:,1:2,:].squeeze().view(1, 1, -1), 3)
            out_1p2d_ma = torch_moving_average_f(out_1p2d[:,0:1,:].squeeze().view(1, 1, -1), 3)

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs[0,0].plot(testing_x_sx[:, 0:1, :].squeeze().detach().numpy(), depth_output, color='lightsteelblue', label=f"step 0")
        axs[0,0].plot(test_profile_d.detach().numpy(), depth_output, color='cornflowerblue', label=f"step 1")
        axs[0,0].plot(test_profile_2d.detach().numpy(), depth_output, color='royalblue', label=f"step 2")
        axs[0,0].plot(testing_x_sx[:, 3:, :].squeeze().detach().numpy(), depth_output, color='midnightblue', label=f"step 3")
        axs[0,0].invert_yaxis()
        axs[0,0].set_title('I0-I1 trend')
        axs[0,1].plot(testing_x_dx[:, 0:1, :].squeeze().detach().numpy(), depth_output, color='lightsteelblue', label=f"step 0")
        axs[0,1].plot(test_profile_1_d.detach().numpy(), depth_output, color='cornflowerblue', label=f"step 1")
        axs[0,1].plot(test_profile_1_2d.detach().numpy(), depth_output, color='royalblue', label=f"step 2")
        axs[0,1].plot(testing_x_dx[:, 3:, :].squeeze().detach().numpy(), depth_output, color='midnightblue', label=f"step 3")
        axs[0,1].invert_yaxis()
        axs[0,1].set_title('I1-I2 trend')

        axs[1,0].plot(testing_x_sx[:, 0:1, :].squeeze().detach().numpy(), depth_output, color='lightsteelblue', label=f"step 0")
        axs[1,0].plot(testing_x_sx[:, 1:2, :].squeeze().detach().numpy(), depth_output, color='lightcoral', label=f"step 1")
        axs[1,0].plot(testing_x_sx[:, 2:3, :].squeeze().detach().numpy(), depth_output, color='firebrick', label=f"step 2")
        axs[1,0].plot(testing_x_sx[:, 3:, :].squeeze().detach().numpy(), depth_output, color='midnightblue', label=f"step 3")
        axs[1,0].invert_yaxis()
        axs[1,0].set_title('I0-I1 observed trend')
        axs[1,1].plot(testing_x_dx[:, 0:1, :].squeeze().detach().numpy(), depth_output, color='lightsteelblue', label=f"step 0")
        axs[1,1].plot(testing_x_dx[:, 1:2, :].squeeze().detach().numpy(), depth_output, color='lightcoral', label=f"step 1")
        axs[1,1].plot(testing_x_dx[:, 2:3, :].squeeze().detach().numpy(), depth_output, color='firebrick', label=f"step 2")
        axs[1,1].plot(testing_x_dx[:, 3:, :].squeeze().detach().numpy(), depth_output, color='midnightblue', label=f"step 3")
        axs[1,1].invert_yaxis()
        axs[1,1].set_title('I1-I2 observed trend')

        plt.legend()
        plt.savefig(dir_profile + f"/profile_{var[:,0:1, 0:1].item()}.png")
        plt.close()