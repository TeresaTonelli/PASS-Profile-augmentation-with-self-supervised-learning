import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from itertools import islice

import seaborn as sns

from NN_architecture import UNet1D, MLP_1D_mode, UNet_1D_mode, UNet_2D_mode
from utils_plots import from_loader_data_to_train_data_f, ssim_plot, mean_ssim_plot, get_evaluation_model_0D, get_evaluation_model_1D, get_evaluation_model_2D


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


def plot_profiles_f(ds, dir, variable, exp_mode, multiple_GPUs = False, ep=100, train=False, mov_average=False, split_timesteps=False, paper_mode=False):
    dir_model = dir + "/model"
    if train == False:
        dir_profile = dir + "/profile_test_paper" if paper_mode == True else dir + "/profile_test"
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

    path_model_conv = dir_model + "/model_" + str(ep) + ".pt"

    models = {"mode_0D": UNet1D, "mode_1D": UNet_1D_mode, "mode_2D": UNet_2D_mode} 
    model = models[exp_mode]()
    model.load_state_dict(torch.load(path_model_conv, map_location=device))
    model.eval()
    if exp_mode == "mode_1D":
        path_model_mlp = dir_model + "/model_mlp_" + str(ep) + ".pt"
        model_MLP = MLP_1D_mode()
        model_MLP.load_state_dict(torch.load(path_model_mlp, map_location=device))
        model_MLP.eval()


    for sample in list(islice(ds, 200)):
        var = sample[1]

        if exp_mode == "mode_0D":
            test_profile_d, test_profile_2d, test_profile_1, test_profile_1_d, test_profile_1_2d, out_2d, out_1p2d = get_evaluation_model_0D(exp_mode, model, sample)

        elif exp_mode == "mode_1D":
            test_profile_d, test_profile_2d, test_profile_1, test_profile_1_d, test_profile_1_2d, out_2d, out_1p2d = get_evaluation_model_1D(exp_mode, model, model_MLP, sample)
        
        elif exp_mode == "mode_2D":
            test_profile_d, test_profile_2d, test_profile_1, test_profile_1_d, test_profile_1_2d, out_2d, out_1p2d = get_evaluation_model_2D(exp_mode, model, sample)

        depth_output = np.linspace(0, max_pres-1, len(test_profile_1.detach().numpy())) #* 5  #prima era [0,0,:]
        depth_variable = np.linspace(0, max_pres-1, len(test_profile_1.detach().numpy()))    #prima era [0,0,:]

        if mov_average == True:
            test_profile_d = torch_moving_average_f(test_profile_d.view(1, 1, -1), 3)
            test_profile_2d = torch_moving_average_f(test_profile_2d.view(1, 1, -1), 3)
            test_profile_1= torch_moving_average_f(test_profile_1.view(1, 1, -1), 3)
            test_profile_1_d= torch_moving_average_f(test_profile_1_d.view(1, 1, -1), 3)
            test_profile_1_2d= torch_moving_average_f(test_profile_1_2d.view(1, 1, -1), 3)
            out_2d_ma= torch_moving_average_f(out_2d[:,1:2,:].squeeze().view(1, 1, -1), 3)
            out_1p2d_ma = torch_moving_average_f(out_1p2d[:,0:1,:].squeeze().view(1, 1, -1), 3)

        if paper_mode == False:
            fig, axs = plt.subplots(2, 3, figsize=(15, 12))
            axs[0, 0].plot(test_profile_d.detach().numpy(), depth_output, color='blue', label=f"generated {variable}")
            axs[0, 0].plot(var[:,1:2,:].squeeze().detach().numpy(), depth_output, color='red', label=f"measured {variable}")
            axs[0, 0].invert_yaxis()
            axs[0, 0].legend(loc="lower left")
            axs[0, 0].set_title(r"$x_{i+{\frac{\delta t}{3}}}$", fontsize=16)
            axs[0, 0].set_xlabel(f"{variable}" r"[$mmol \ m^{-3}$]")
            axs[0, 0].set_ylabel(r"depth [$m$]")
            axs[0, 1].plot(test_profile_2d.detach().numpy(), depth_output, color='blue', label=f"generated {variable} ")
            axs[0, 1].plot(var[:,2:3,:].squeeze().detach().numpy(), depth_output, color='red', label=f"measured {variable}")
            axs[0, 1].invert_yaxis()
            axs[0, 1].legend(loc="lower left")
            axs[0, 1].set_title(r"$x_{i+{\frac{2\delta t}{3}}}$", fontsize=16)
            axs[0, 1].set_xlabel(f"{variable}" r"[$mmol \ m^{-3}$]")
            axs[0, 1].set_ylabel(r"depth [$m$]")
            axs[0, 2].plot(test_profile_1.detach().numpy(), depth_output, color='blue', label=f"generated {variable}")
            axs[0, 2].plot(var[:,3:4,:].squeeze().detach().numpy(), depth_output, color='red', label=f"measured {variable}")
            axs[0, 2].invert_yaxis()
            axs[0, 2].legend(loc="lower left")
            axs[0, 2].set_title(r"$x_{i+{\delta t}}$", fontsize=16)
            axs[0, 2].set_xlabel(f"{variable}" r"[$mmol \ m^{-3}$]")
            axs[0, 2].set_ylabel(r"depth [$m$]")
            axs[1, 0].plot(test_profile_1_d.detach().numpy(), depth_output, color='blue', label=f"generated {variable}")
            axs[1, 0].plot(var[:,4:5,:].squeeze().detach().numpy(), depth_output, color='red', label=f"measured {variable}")
            axs[1, 0].invert_yaxis()
            axs[1, 0].legend(loc="lower left")
            axs[1, 0].set_title(r"$x_{i+{\frac{4 \delta t}{3}}}$", fontsize=16)
            axs[1, 0].set_xlabel(f"{variable}" r"[$mmol \ m^{-3}$]")
            axs[1, 0].set_ylabel(r"depth [$m$]")
            axs[1, 1].plot(test_profile_1_2d.detach().numpy(), depth_output, color='blue', label=f"generated {variable}")
            axs[1, 1].plot(var[:,5:6,:].squeeze().detach().numpy(), depth_output, color='red', label=f"measured {variable}")
            axs[1, 1].invert_yaxis()
            axs[1, 1].legend(loc="lower left")
            axs[1, 1].set_title(r"$x_{i+{\frac{5 \delta t}{3}}}$", fontsize=16)
            axs[1, 1].set_xlabel(f"{variable}" r"[$mmol \ m^{-3}$]")
            axs[1, 1].set_ylabel(r"depth [$m$]")
            axs[1, 2].plot(out_2d[:,1:2,:].squeeze().detach().numpy(), depth_output, color='green', label=f"generated tc1 {variable}")
            axs[1, 2].plot(out_1p2d[:,0:1,:].squeeze().detach().numpy(), depth_output, color='orange', label=f"generated tc2 {variable}")
            axs[1, 2].plot(var[:,3:4,:].squeeze().detach().numpy(), depth_output, color='red', label=f"measured {variable}")
            axs[1, 2].invert_yaxis()
            axs[1, 2].legend(loc="lower left")
            axs[1, 2].set_title(r"$x_{i+{\delta t}}$" ": dual cycle", fontsize=16)
            axs[1, 2].set_xlabel(f"{variable}" r"[$mmol \ m^{-3}$]")
            axs[1, 2].set_ylabel(r"depth [$m$]")

        elif paper_mode == True:
            sns.set_theme(context='paper', style='whitegrid', font='sans-serif', font_scale=2.5, color_codes=True, rc=None)
            fig, axs = plt.subplots(1, 6, figsize=(45, 10))
            axs[0].plot(test_profile_d.detach().numpy(), depth_output, linewidth=3.5, color='blue', label=f"generated {variable}")
            axs[0].plot(var[:,1:2,:].squeeze().detach().numpy(), depth_output, linewidth=3.5, color='red', label=f"measured {variable}")
            axs[0].invert_yaxis()
            #axs[0].legend(loc="lower left")
            axs[0].set_title(r"$x_{t_0+{\frac{\delta t}{3}}}$", fontsize=36)
            axs[0].set_xlabel(f"{variable}" r"[$mg \ m^{-3}$]")
            axs[0].set_ylabel(r"depth [$m$]")
            axs[1].plot(test_profile_2d.detach().numpy(), depth_output, linewidth=3.5, color='blue', label=f"generated {variable} ")
            axs[1].plot(var[:,2:3,:].squeeze().detach().numpy(), depth_output, linewidth=3.5, color='red', label=f"measured {variable}")
            axs[1].invert_yaxis()
            #axs[1].legend(loc="lower left")
            axs[1].set_title(r"$x_{t_0+{\frac{2\delta t}{3}}}$", fontsize=36)
            axs[1].set_xlabel(f"{variable}" r"[$mg \ m^{-3}$]")
            #axs[1].set_ylabel(r"depth [$m$]")
            axs[2].plot(test_profile_1.detach().numpy(), depth_output, linewidth=3.5, color='blue', label=f"generated {variable}")
            axs[2].plot(var[:,3:4,:].squeeze().detach().numpy(), depth_output, linewidth=3.5, color='red', label=f"measured {variable}")
            axs[2].invert_yaxis()
            #axs[2].legend(loc="lower left")
            axs[2].set_title(r"$x_{t_0+{\delta t}}$", fontsize=36)
            axs[2].set_xlabel(f"{variable}" r"[$mg \ m^{-3}$]")
            #axs[2].set_ylabel(r"depth [$m$]")
            axs[3].plot(test_profile_1_d.detach().numpy(), depth_output, linewidth=3.5, color='blue', label=f"generated {variable}")
            axs[3].plot(var[:,4:5,:].squeeze().detach().numpy(), depth_output, linewidth=3.5, color='red', label=f"measured {variable}")
            axs[3].invert_yaxis()
            #axs[3].legend(loc="lower left")
            axs[3].set_title(r"$x_{t_0+{\frac{4 \delta t}{3}}}$", fontsize=36)
            axs[3].set_xlabel(f"{variable}" r"[$mg \ m^{-3}$]")
            #axs[3].set_ylabel(r"depth [$m$]")
            axs[4].plot(test_profile_1_2d.detach().numpy(), depth_output, linewidth=3.5, color='blue', label=f"generated {variable}")
            axs[4].plot(var[:,5:6,:].squeeze().detach().numpy(), depth_output, linewidth=3.5, color='red', label=f"measured {variable}")
            axs[4].invert_yaxis()
            #axs[4].legend(loc="lower left")
            axs[4].set_title(r"$x_{t_0+{\frac{5 \delta t}{3}}}$", fontsize=36)
            axs[4].set_xlabel(f"{variable}" r"[$mg \ m^{-3}$]")
            #axs[4].set_ylabel(r"depth [$m$]")
            axs[5].plot(out_2d[:,1:2,:].squeeze().detach().numpy(), depth_output, linewidth=3.5, linestyle='--', color='green', label=f"generated tc1 {variable}")
            axs[5].plot(out_1p2d[:,0:1,:].squeeze().detach().numpy(), depth_output, linewidth=3.5, linestyle='--', color='orange', label=f"generated tc2 {variable}")
            axs[5].plot(var[:,3:4,:].squeeze().detach().numpy(), depth_output, linewidth=3.5, color='red', label=f"measured {variable}")
            axs[5].invert_yaxis()
            #axs[5].legend(loc="lower left")
            axs[5].set_title(r"$x_{t_0+{\delta t}}$" ": dual cycle", fontsize=30)
            axs[5].set_xlabel(f"{variable}" r"[$mg \ m^{-3}$]")
            #axs[5].set_ylabel(r"depth [$m$]")

        if split_timesteps == True:
            days = sample[2][0, :, 0].squeeze() 
            days = [(d * 365) / (2 * np.pi) for d in days]
            d = int(days[1].item() - days[0].item())
            dir_timestep = dir_profile + f"/day_{str(int(d))}" 
            if not os.path.exists(dir_timestep):
                os.mkdir(path=dir_timestep)
            plt.savefig(dir_timestep + f"/profile_{var[:,0:1, 0:1].item()}.png", dpi=600)
            plt.close()

        else:
            plt.savefig(dir_profile + f"/profile_{var[:,0:1, 0:1].item()}.png", dpi=600)
            plt.close()


def plot_NN_intermediate_profiles(ds, dir, variable, ep=100, mov_average=False, split_timesteps=False):
    dir_model = dir + "/model"
    dir_profile = dir + "/NN_intermediate_profile"
    if not os.path.exists(dir_profile):
        os.mkdir(path=dir_profile)
    dir_info = dir + "/info.csv"
    info_df = pd.read_csv(dir_info)
    max_pres = 200    
    interval = 1      
    path_model_conv = dir_model + "/model_" + str(ep) + ".pt"
    model = UNet1D() #.to(device)   #AGGIUNTO ORA QUESTO
    model.load_state_dict(torch.load(path_model_conv))
    model.eval()

    for sample in list(islice(ds, 50)):
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

        fig, axs = plt.subplots(2, 3, figsize=(14, 10))
        axs[0,0].plot(testing_x_sx[:, 0:1, :].squeeze().detach().numpy(), depth_output, color='lightsteelblue', label=f"step 0")
        axs[0,0].plot(test_profile_d.detach().numpy(), depth_output, color='cornflowerblue', label=f"step 1")
        axs[0,0].plot(test_profile_2d.detach().numpy(), depth_output, color='royalblue', label=f"step 2")
        axs[0,0].plot(testing_x_sx[:, 3:, :].squeeze().detach().numpy(), depth_output, color='midnightblue', label=f"step 3")
        axs[0,0].invert_yaxis()
        axs[0,0].set_title('I0-I1 trend')
        axs[0,0].legend(loc="lower left")
        axs[1,0].plot(testing_x_dx[:, 0:1, :].squeeze().detach().numpy(), depth_output, color='lightsteelblue', label=f"step 0")
        axs[1,0].plot(test_profile_1_d.detach().numpy(), depth_output, color='cornflowerblue', label=f"step 1")
        axs[1,0].plot(test_profile_1_2d.detach().numpy(), depth_output, color='royalblue', label=f"step 2")
        axs[1,0].plot(testing_x_dx[:, 3:, :].squeeze().detach().numpy(), depth_output, color='midnightblue', label=f"step 3")
        axs[1,0].invert_yaxis()
        axs[1,0].set_title('I1-I2 trend')
        axs[1,0].legend(loc="lower left")

        axs[0,1].plot(testing_x_sx[:, 0:1, :].squeeze().detach().numpy(), depth_output, color='lightsteelblue', label=f"step 0")
        axs[0,1].plot(testing_x_sx[:, 1:2, :].squeeze().detach().numpy(), depth_output, color='lightcoral', label=f"step 1")
        axs[0,1].plot(testing_x_sx[:, 2:3, :].squeeze().detach().numpy(), depth_output, color='firebrick', label=f"step 2")
        axs[0,1].plot(testing_x_sx[:, 3:, :].squeeze().detach().numpy(), depth_output, color='midnightblue', label=f"step 3")
        axs[0,1].invert_yaxis()
        axs[0,1].set_title('I0-I1 observed trend')
        axs[0,1].legend(loc="lower left")
        axs[1,1].plot(testing_x_dx[:, 0:1, :].squeeze().detach().numpy(), depth_output, color='lightsteelblue', label=f"step 0")
        axs[1,1].plot(testing_x_dx[:, 1:2, :].squeeze().detach().numpy(), depth_output, color='lightcoral', label=f"step 1")
        axs[1,1].plot(testing_x_dx[:, 2:3, :].squeeze().detach().numpy(), depth_output, color='firebrick', label=f"step 2")
        axs[1,1].plot(testing_x_dx[:, 3:, :].squeeze().detach().numpy(), depth_output, color='midnightblue', label=f"step 3")
        axs[1,1].invert_yaxis()
        axs[1,1].set_title('I1-I2 observed trend')
        axs[1,1].legend(loc="lower left")

        axs[0,2].plot(testing_x_sx[:, 0:1, :].squeeze().detach().numpy(), depth_output, color='hotpink', label=f"step 0")
        axs[0,2].plot(model(testing_x_sx[:, [0,3], :])[:, 0:1, :].squeeze().detach().numpy(), depth_output, color='magenta', label=f"step 1")
        axs[0,2].plot(model(testing_x_sx[:, [0,3], :])[:, 1:2, :].squeeze().detach().numpy(), depth_output, color='darkviolet', label=f"step 2")
        axs[0,2].plot(model(torch.cat((testing_x_sx[:, 1:2, :], testing_x_dx[:, 1:2, :]), 1))[:, 0:1, :].squeeze().detach().numpy(), depth_output, color='darkviolet', label=f"step 2")
        axs[0,2].plot(model(torch.cat((testing_x_sx[:, 1:2, :], testing_x_dx[:, 1:2, :]), 1))[:, 1:2, :].squeeze().detach().numpy(), depth_output, color='rebeccapurple', label=f"step 3")
        axs[0,2].plot(testing_x_sx[:, 3:, :].squeeze().detach().numpy(), depth_output, color='rebeccapurple', label=f"step 3")
        axs[0,2].invert_yaxis()
        axs[0,2].set_title('I0-I1 observed trend')
        axs[0,2].legend(loc="lower left")
        axs[1,2].plot(testing_x_dx[:, 0:1, :].squeeze().detach().numpy(), depth_output, color='hotpink', label=f"step 0")
        axs[1,2].plot(model(testing_x_dx[:, [0,3], :])[:, 0:1, :].squeeze().detach().numpy(), depth_output, color='magenta', label=f"step 1")
        axs[1,2].plot(model(testing_x_dx[:, [0,3], :])[:, 1:2, :].squeeze().detach().numpy(), depth_output, color='darkviolet', label=f"step 2")
        axs[1,2].plot(model(torch.cat((testing_x_sx[:, 2:3, :], testing_x_dx[:, 2:3, :]), 1))[:, 0:1, :].squeeze().detach().numpy(), depth_output, color='darkviolet', label=f"step 2")
        axs[1,2].plot(model(torch.cat((testing_x_sx[:, 2:3, :], testing_x_dx[:, 2:3, :]), 1))[:, 1:2, :].squeeze().detach().numpy(), depth_output, color='rebeccapurple', label=f"step 3")
        axs[1,2].plot(testing_x_dx[:, 3:, :].squeeze().detach().numpy(), depth_output, color='rebeccapurple', label=f"step 3")
        axs[1,2].invert_yaxis()
        axs[1,2].set_title('I1-I2 observed trend')
        axs[1,2].legend(loc="lower left")

        #plt.savefig(dir_profile + f"/profile_{var[:,0:1, 0:1].item()}.png")
        #plt.close()

        #part for the Pearson correlation coefficient
        with open(dir_profile + "/pearson_output.txt", "a") as f:
            f.write(f"/profile_{var[:,0:1, 0:1].item()}\n")
            for j in range(var.shape[1]-1):
                pearson_coeff = np.corrcoef(var[:,j:j+1,:].squeeze().numpy(), var[:,j+1:j+2,:].squeeze().numpy())
                f.write(str(pearson_coeff[0,1]) + "\n")

        with open(dir_profile + "/SSIM_output.txt", "a") as f:
            f.write(f"/profile_{var[:,0:1, 0:1].item()}\n")
            St, Ssim_t, mu, sigma = mean_ssim_plot(var, data_range=0.4)
            f.write("mu = " + str(mu) + "\n")
            f.write("sigma = " + str(sigma) + "\n")
            for t in range(Ssim_t.shape[0]):
                f.write(str(Ssim_t[t]) + "\n")

        if split_timesteps == True:
            days = sample[2][0, :, 0].squeeze() 
            days = [(d * 365) / (2 * np.pi) for d in days]
            d = int(days[1].item() - days[0].item())
            dir_timestep = dir_profile + f"/day_{str(int(d))}" 
            if not os.path.exists(dir_timestep):
                os.mkdir(path=dir_timestep)
            plt.savefig(dir_timestep + f"/profile_{var[:,0:1, 0:1].item()}.png")
            plt.close()

        else:
            plt.savefig(dir_profile + f"/profile_{var[:,0:1, 0:1].item()}.png")
            plt.close()





def linear_interpolation_comparison(ds, dir, variable, exp_mode, ep=100, mov_average=False, split_timesteps=False, paper_mode=False):
    dir_model = dir + "/model"
    dir_profile = dir + "/linear_interpolation_comparison" if paper_mode == False else dir + "/linear_interpolation_comparison_paper"
    dir_single_comparison = dir + "/linear_interpolation_single_comparison" if paper_mode == False else dir + "/linear_interpolation_single_comparison_paper"
    os.makedirs(dir_profile, exist_ok=True)
    os.makedirs(dir_single_comparison, exist_ok=True)
    for j in [1,2,4,5]:
        dir_single_comparison_j = dir_single_comparison + f"/prediction_{j}"
        os.makedirs(dir_single_comparison_j, exist_ok=True)
    dir_info = dir + "/info.csv"
    info_df = pd.read_csv(dir_info)
    max_pres = 200    
    interval = 1      

    path_model_conv = dir_model + "/model_" + str(ep) + ".pt"
    models = {"mode_0D": UNet1D, "mode_1D": UNet_1D_mode, "mode_2D":UNet_2D_mode} 
    model = models[exp_mode]()
    model.load_state_dict(torch.load(path_model_conv, map_location=device))
    model.eval()
    if exp_mode == "mode_1D":
        path_model_mlp = dir_model + "/model_mlp_" + str(ep) + ".pt"
        model_MLP = MLP_1D_mode()
        model_MLP.load_state_dict(torch.load(path_model_mlp, map_location=device))
        model_MLP.eval()

    for sample in list(islice(ds, 200)):
        var = sample[1]
        testing_x_tmp = from_loader_data_to_train_data_f(sample, exp_mode)
        testing_x_sx = testing_x_tmp[0]
        testing_x_dx = testing_x_tmp[1]

        if exp_mode == "mode_0D":
            test_profile_d, test_profile_2d, test_profile_1, test_profile_1_d, test_profile_1_2d, out_2d, out_1p2d = get_evaluation_model_0D(exp_mode, model, sample)

        elif exp_mode == "mode_1D":
            test_profile_d, test_profile_2d, test_profile_1, test_profile_1_d, test_profile_1_2d, out_2d, out_1p2d = get_evaluation_model_1D(exp_mode, model, model_MLP, sample)
         
        elif exp_mode == "mode_2D":
            test_profile_d, test_profile_2d, test_profile_1, test_profile_1_d, test_profile_1_2d, out_2d, out_1p2d = get_evaluation_model_2D(exp_mode, model, sample)

        #out_d = model(testing_x_sx[:, [0,3], :])  
        #out_1pd = model(testing_x_dx[:, [0,3], :]) 
        #out_2d = model(torch.cat((out_d[:, 0:1, :], out_1pd[:, 0:1, :]), 1))
        #out_1p2d = model(torch.cat((out_d[:, 1:2, :], out_1pd[:, 1:2, :]), 1))

        #test_profile_d = out_d[:,0:1,:].squeeze()
        #test_profile_1_2d = out_1pd[:,1:2,:].squeeze()
        #test_profile_2d = ((out_d[:,1:2,:] + out_2d[:,0:1,:])/2).squeeze()
        #test_profile_1_d = ((out_1pd[:,0:1,:] + out_1p2d[:,1:2,:])/2).squeeze()
        #test_profile_1 = ((testing_x_sx[:,-1:,:]+out_2d[:,1:2,:]+out_1p2d[:,0:1,:])/3).squeeze()

        out_1_linear = (2/3)*testing_x_sx[:, 0:1, :] + (1/3)*testing_x_sx[:, 3:, :]
        out_2_linear = (1/3)*testing_x_sx[:, 0:1, :] + (2/3)*testing_x_sx[:, 3:, :]
        out_4_linear = (2/3)*testing_x_dx[:, 0:1, :] + (1/3)*testing_x_dx[:, 3:, :]
        out_5_linear = (1/3)*testing_x_dx[:, 0:1, :] + (2/3)*testing_x_dx[:, 3:, :]

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

        if paper_mode == False:
            fig, axs = plt.subplots(1, 2, figsize=(14, 10))
            axs[0].plot(testing_x_sx[:, 0:1, :].squeeze().detach().numpy(), depth_output, color='lightsteelblue', label=f"step 0")
            axs[0].plot(test_profile_d.detach().numpy(), depth_output, color='cornflowerblue', label=f"step 1")
            axs[0].plot(test_profile_2d.detach().numpy(), depth_output, color='royalblue', label=f"step 2")
            axs[0].plot(testing_x_sx[:, 3:, :].squeeze().detach().numpy(), depth_output, color='midnightblue', label=f"step 3")
            axs[0].plot(out_1_linear.squeeze(), depth_output, color='lightcoral', linestyle='--', label=f"linear interpolation 1")
            axs[0].plot(out_2_linear.squeeze(), depth_output, color='firebrick', linestyle='--', label=f"linear interpolation 2")
            axs[0].invert_yaxis()
            axs[0].set_title('step 0-3 trend')
            axs[0].legend(loc="lower left")
            axs[1].plot(testing_x_dx[:, 0:1, :].squeeze().detach().numpy(), depth_output, color='lightsteelblue', label=f"step 3")
            axs[1].plot(test_profile_1_d.detach().numpy(), depth_output, color='cornflowerblue', label=f"step 4")
            axs[1].plot(test_profile_1_2d.detach().numpy(), depth_output, color='royalblue', label=f"step 5")
            axs[1].plot(testing_x_dx[:, 3:, :].squeeze().detach().numpy(), depth_output, color='midnightblue', label=f"step 6")
            axs[1].plot(out_4_linear.squeeze(), depth_output, color='lightcoral', linestyle='--', label=f"linear interpolation 4")
            axs[1].plot(out_5_linear.squeeze(), depth_output, color='firebrick', linestyle='--', label=f"linear interpolation 5")
            axs[1].invert_yaxis()
            axs[1].set_title('step 3-6 trend')
            axs[1].legend(loc="lower left")
        elif paper_mode == True:
            sns.set_theme(context='paper', style='whitegrid', font='sans-serif', font_scale=2.5, color_codes=True, rc=None)
            fig, axs = plt.subplots(1, 2, figsize=(14, 10))
            axs[0].plot(testing_x_sx[:, 0:1, :].squeeze().detach().numpy(), depth_output, linewidth=2.5, color='lightsteelblue', label=f"step 0")
            axs[0].plot(test_profile_d.detach().numpy(), depth_output, linewidth=2.5, color='cornflowerblue', label=f"step 1")
            axs[0].plot(test_profile_2d.detach().numpy(), depth_output, linewidth=2.5, color='royalblue', label=f"step 2")
            axs[0].plot(testing_x_sx[:, 3:, :].squeeze().detach().numpy(), depth_output, linewidth=2.5, color='midnightblue', label=f"step 3")
            axs[0].plot(out_1_linear.squeeze(), depth_output, linewidth=2.5, color='lightcoral', linestyle='--', label=f"linear interpolation 1")
            axs[0].plot(out_2_linear.squeeze(), depth_output, linewidth=2.5, color='firebrick', linestyle='--', label=f"linear interpolation 2")
            axs[0].invert_yaxis()
            axs[0].set_title(r"[$t_0, t_0+\delta t$] step")
            axs[0].set_xlabel(f"{variable}" r"[$mg \ m^{-3}$]")
            axs[0].set_ylabel(r"depth [$m$]")
            #axs[0].legend(loc="lower left")
            axs[1].plot(testing_x_dx[:, 0:1, :].squeeze().detach().numpy(), depth_output, linewidth=2.5, color='lightsteelblue', label=f"step 3")
            axs[1].plot(test_profile_1_d.detach().numpy(), depth_output, linewidth=2.5, color='cornflowerblue', label=f"step 4")
            axs[1].plot(test_profile_1_2d.detach().numpy(), depth_output, linewidth=2.5, color='royalblue', label=f"step 5")
            axs[1].plot(testing_x_dx[:, 3:, :].squeeze().detach().numpy(), depth_output, linewidth=2.5, color='midnightblue', label=f"step 6")
            axs[1].plot(out_4_linear.squeeze(), depth_output, color='lightcoral', linewidth=2.5, linestyle='--', label=f"linear interpolation 4")
            axs[1].plot(out_5_linear.squeeze(), depth_output, color='firebrick', linewidth=2.5, linestyle='--', label=f"linear interpolation 5")
            axs[1].invert_yaxis()
            axs[1].set_title(r"[$t_0+\delta t, t_0+2\delta t$] step")
            axs[1].set_xlabel(f"{variable}" r"[$mg \ m^{-3}$]")
            #axs[1].legend(loc="lower left")

        if split_timesteps == True:
            days = sample[2][0, :, 0].squeeze() 
            days = [(d * 365) / (2 * np.pi) for d in days]
            d = int(days[1].item() - days[0].item())
            dir_timestep = dir_profile + f"/day_{str(int(d))}" 
            if not os.path.exists(dir_timestep):
                os.mkdir(path=dir_timestep)
            plt.savefig(dir_timestep + f"/profile_{var[:,0:1, 0:1].item()}.png")
            plt.close()

        else:
            plt.savefig(dir_profile + f"/profile_{var[:,0:1, 0:1].item()}.png")
            plt.close()


        #single comparison 1
        save_dir_indexes = ["1", "2", "4", "5"]
        outputs_1DUNet = [test_profile_d, test_profile_2d, test_profile_1_d, test_profile_1_2d]
        outputs_floats = [testing_x_sx[:, 1:2, :], testing_x_sx[:, 2:3, :], testing_x_dx[:, 1:2, :], testing_x_dx[:, 2:3, :]]
        outputs_linear = [out_1_linear, out_2_linear, out_4_linear, out_5_linear]

        if paper_mode == False:
            for l in range(len(outputs_1DUNet)):
                sns.set_theme(context='paper', style='whitegrid', font='sans-serif', font_scale=2.5, color_codes=True, rc=None)
                plt.figure(figsize=(8, 10))
                plt.plot(outputs_1DUNet[l].detach().numpy(), depth_output, linewidth=2.5, color='blue', label=f"1DUnet 1")
                plt.plot(outputs_floats[l].squeeze().detach().numpy(), depth_output, linewidth=2.5, color='red', label=f"ARGO-float 1")
                plt.plot(outputs_linear[l].squeeze(), depth_output, linewidth=2.5, color='orange', label=f"Linear interpolation 1")
                plt.gca().invert_yaxis()
                plt.xlabel(f"{variable}" r"[$mg \ m^{-3}$]")
                plt.ylabel(r"depth [$m$]")
                plt.savefig(dir_single_comparison + f"/prediction_{save_dir_indexes[l]}/profile_{var[:,0:1, 0:1].item()}.png")
                plt.close()

        if paper_mode == True:
            sns.set_theme(context='paper', style='whitegrid', font='sans-serif', font_scale=2.5, color_codes=True, rc=None)
            fig, axs = plt.subplots(1, 4, figsize=(30, 10))
            axs[0].plot(outputs_1DUNet[0].detach().numpy(), depth_output, linewidth=2.5, color='blue', label=f"1DUnet 1")
            axs[0].plot(outputs_floats[0].squeeze().detach().numpy(), depth_output, linewidth=2.5, color='red', label=f"ARGO-float 1")
            axs[0].plot(outputs_linear[0].squeeze(), depth_output, linewidth=2.5, color='orange', label=f"Linear interpolation 1")
            axs[0].invert_yaxis()
            axs[0].set_title(r"$x_{t_0+{\frac{\delta t}{3}}}$", fontsize=36)
            axs[0].set_xlabel(f"{variable}" r"[$mg \ m^{-3}$]")
            axs[0].set_ylabel(r"depth [$m$]")
            axs[1].plot(outputs_1DUNet[1].detach().numpy(), depth_output, linewidth=2.5, color='blue', label=f"1DUnet 1")
            axs[1].plot(outputs_floats[1].squeeze().detach().numpy(), depth_output, linewidth=2.5, color='red', label=f"ARGO-float 1")
            axs[1].plot(outputs_linear[1].squeeze(), depth_output, linewidth=2.5, color='orange', label=f"Linear interpolation 1")
            axs[1].invert_yaxis()
            axs[1].set_title(r"$x_{t_0+{\frac{2\delta t}{3}}}$", fontsize=36)
            axs[1].set_xlabel(f"{variable}" r"[$mg \ m^{-3}$]")
            axs[2].plot(outputs_1DUNet[2].detach().numpy(), depth_output, linewidth=2.5, color='blue', label=f"1DUnet 1")
            axs[2].plot(outputs_floats[2].squeeze().detach().numpy(), depth_output, linewidth=2.5, color='red', label=f"ARGO-float 1")
            axs[2].plot(outputs_linear[2].squeeze(), depth_output, linewidth=2.5, color='orange', label=f"Linear interpolation 1")
            axs[2].invert_yaxis()
            axs[2].set_title(r"$x_{t_0+{\frac{4\delta t}{3}}}$", fontsize=36)
            axs[2].set_xlabel(f"{variable}" r"[$mg \ m^{-3}$]")
            axs[3].plot(outputs_1DUNet[3].detach().numpy(), depth_output, linewidth=2.5, color='blue', label=f"1DUnet 1")
            axs[3].plot(outputs_floats[3].squeeze().detach().numpy(), depth_output, linewidth=2.5, color='red', label=f"ARGO-float 1")
            axs[3].plot(outputs_linear[3].squeeze(), depth_output, linewidth=2.5, color='orange', label=f"Linear interpolation 1")
            axs[3].invert_yaxis()
            axs[3].set_title(r"$x_{t_0+{\frac{5\delta t}{3}}}$", fontsize=36)
            axs[3].set_xlabel(f"{variable}" r"[$mg \ m^{-3}$]")
            plt.savefig(dir_single_comparison + f"/sequence_{var[:,0:1, 0:1].item()}.png", dpi=400)
            plt.close()

