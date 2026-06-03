import os
import torch 
import torch.nn.functional as F


def read_losses_median_f(losses_txt_file):
    losses_medians = []
    with open(losses_txt_file, 'r') as file:
        for line in file:
            values = eval(line.strip())
            if isinstance(values, list) and len(values) >= 3:
                losses_medians.append(values[2]) 
            else:
                raise ValueError(f"Invalid line or not enough elements: {line}")
    return losses_medians


def ssim_plot(x,y,data_range, window_size=11):
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    padding = window_size // 2
    mu_x = F.avg_pool1d(x, window_size, stride=1, padding=padding)
    mu_y = F.avg_pool1d(y, window_size, stride=1, padding=padding)
    sigma_x = F.avg_pool1d(x * x, window_size, 1, padding) - mu_x ** 2
    sigma_y = F.avg_pool1d(y * y, window_size, 1, padding) - mu_y ** 2
    sigma_xy = F.avg_pool1d(x * y, window_size, 1, padding) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
    return ssim_map.mean()


def mean_ssim_plot(var_tensor, data_range, normalize=False):
    if normalize == True: 
        data_range = 1.0
    if normalize == False:
        my_var_tensor = var_tensor
    if normalize == True:
        var_tensor_norm = var_tensor.clone()    #questo poi sarà da modificare in caso
        for t in range(var_tensor.shape[1]-1):
            m = torch.mean(var_tensor[:, t:t+1, :])
            std = torch.std(var_tensor[:, t:t+1, :])
            #m = var_tensor[:, t:t+1, :].mean(dim=-1, keepdim=True)
            #std = var_tensor[:, t:t+1, :].std(dim=-1, unbiased=False, keepdim=True)
            #std = torch.clamp(std, min=1e-6)
            var_tensor_norm[:, t:t+1,:] = (var_tensor_norm[:, t:t+1,:]-m)/std
        my_var_tensor = var_tensor_norm
    mean_var = torch.mean(my_var_tensor, dim=1)
    St = torch.zeros(my_var_tensor.shape[1])
    Ssim_t = torch.zeros(my_var_tensor.shape[1]-1)
    for t in range(my_var_tensor.shape[1]):
        St[t] = ssim_plot(my_var_tensor[:,t:t+1,:], mean_var, data_range)
    for t in range(my_var_tensor.shape[1]-1):
        Ssim_t[t] = ssim_plot(my_var_tensor[:,t:t+1,:], my_var_tensor[:, t+1:t+2,:], data_range)
    mu = torch.mean(St)
    sigma = torch.std(St)
    return St, Ssim_t, mu, sigma


def from_loader_data_to_train_data_f(input_data, exp_mode):
    testing_x = input_data[1]
    if exp_mode == "mode_1D":
        testing_surf_x = input_data[2]
    n = testing_x.shape[1]
    l = n//2
    ##Split between training sx and training dx
    testing_x_sx = testing_x[:, 0:l+1, :] 
    testing_x_dx = testing_x[:, l:n+1, :]
    assert (testing_x_sx[:, 3:, :] == testing_x_dx[:, 0:1, :]).all()
    if exp_mode == "mode_1D":
        return testing_x_sx, testing_x_dx, testing_surf_x
    else: 
        return testing_x_sx, testing_x_dx


def get_evaluation_model_0D(exp_mode, model, sample):
    testing_x_tmp = from_loader_data_to_train_data_f(sample, exp_mode)
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

    return test_profile_d, test_profile_2d, test_profile_1, test_profile_1_d, test_profile_1_2d, out_2d, out_1p2d



def get_evaluation_model_1D(exp_mode, model, model_mlp, sample):
    testing_x_tmp = from_loader_data_to_train_data_f(sample, exp_mode)
    testing_x_sx, testing_x_dx, testing_surf_x = testing_x_tmp  
    output_surf = model_mlp(testing_surf_x)
    
    out_d = model(torch.cat((output_surf, testing_x_sx[:, [0,3], :]), 1))  
    out_1pd = model(torch.cat((output_surf, testing_x_dx[:, [0,3], :]), 1)) 
    out_2d = model(torch.cat((output_surf, out_d[:, 0:1, :], out_1pd[:, 0:1, :]), 1))
    out_1p2d = model(torch.cat((output_surf, out_d[:, 1:2, :], out_1pd[:, 1:2, :]), 1))

    test_profile_d = out_d[:,0:1,:].squeeze()
    test_profile_1_2d = out_1pd[:,1:2,:].squeeze()
    test_profile_2d = ((out_d[:,1:2,:] + out_2d[:,0:1,:])/2).squeeze()
    test_profile_1_d = ((out_1pd[:,0:1,:] + out_1p2d[:,1:2,:])/2).squeeze()
    test_profile_1 = ((testing_x_sx[:,-1:,:]+out_2d[:,1:2,:]+out_1p2d[:,0:1,:])/3).squeeze()

    return test_profile_d, test_profile_2d, test_profile_1, test_profile_1_d, test_profile_1_2d, out_2d, out_1p2d




def get_evaluation_model_2D(ds, dir, variable, ep=100, mov_average=False, split_timesteps=False):
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
