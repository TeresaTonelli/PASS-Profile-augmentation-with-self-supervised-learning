import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
from datetime import date
import time
import numpy as np

from errors_evaluation import RMSE_comparison
from NN_architecture import UNet1D
from plots import plot_loss_f, plot_profiles_f, plot_NN_intermediate_profiles, linear_interpolation_comparison
from utils_dataset import FloatDatasetFeat, compute_and_save_normalization_stats_f
from utils_plots import read_losses_median_f
from utils_src import create_directories_f, save_ds_info_f, EMD_loss

print("end installation")


class TrainerFeat:
    def __init__(
        self,
        model_Conv1D: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        save_every: int,
        alpha_smooth_reg: float, 
    ) -> None:
        self.device = device
        self.model_Conv1D = model_Conv1D.to(self.device)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.alpha_smooth_reg = alpha_smooth_reg

    def _data_train_preprocess_f(self, input_data, batch_size):
        training_x = input_data[1]
        n = training_x.shape[1]
        l = n//2
        training_x_sx = training_x[:, 0:l+1, :] 
        training_x_dx = training_x[:, l:n+1, :]
        assert (training_x_sx[:, 3, :] == training_x_dx[:, 0, :]).all()
        return training_x_sx, training_x_dx

    def _run_batch_f(self, list_sources, targets, losses):
        self.optimizer.zero_grad()
        I_01, I_02 = list_sources
        assert (targets == I_01[:, -1:, :]).all()
        assert (targets == I_02[:, 0:1, :]).all() 
        I_d = self.model_Conv1D(I_01[:, [0,3], :])  
        I_1pd = self.model_Conv1D(I_02[:, [0,3], :])
        print("input shape", torch.cat((I_d[:, 0:1, :], I_1pd[:, 0:1, :]), 1).shape)
        I_d2 = self.model_Conv1D(torch.cat((I_d[:, 0:1, :], I_1pd[:, 0:1, :]), 1))    
        I_1pd2 = self.model_Conv1D(torch.cat((I_d[:, 1:2, :], I_1pd[:, 1:2, :]), 1))     
        #loss_1 = torch.nn.functional.mse_loss(I_d2[:, :, :, -1:, 1:2], targets)  # + torch.nn.functional.mse_loss(I_1pd2[:, :, :, -1:, 0:1], targets)) / 2      
        loss_1 = (torch.nn.functional.mse_loss(I_d2[:, 1:2, :], targets) + torch.nn.functional.mse_loss(I_1pd2[:, 0:1, :], targets)) / 2      
        loss_1sd = torch.nn.functional.mse_loss(I_d2[:, 1:2, :], I_1pd2[:, 0:1, :])     
        loss_2 = (torch.nn.functional.mse_loss(I_d[:, 1:2, :], I_d2[:, 0:1, :]) + torch.nn.functional.mse_loss(I_1pd[:, 0:1, :], I_1pd2[:, 1:2, :])) / 2      
        loss_d = torch.nn.functional.mse_loss(I_d[:, 0:1, :], I_01[:, 1:2, :])     
        loss_2d = (torch.nn.functional.mse_loss(I_d[:, 1:2, :], I_01[:, 2:3, :]) + torch.nn.functional.mse_loss(I_d2[:, 0:1, :], I_01[:, 2:3, :])) / 2       
        loss_1pd = (torch.nn.functional.mse_loss(I_1pd[:, 0:1, :], I_02[:, 1:2, :]) + torch.nn.functional.mse_loss(I_1pd2[:, 1:2, :], I_02[:, 1:2, :])) / 2   
        loss_1p2d = torch.nn.functional.mse_loss(I_1pd[:, 1:2, :], I_02[:, 2:3, :])   
        loss_bound = torch.nn.functional.mse_loss(self.model_Conv1D(torch.cat((I_01[:, 0:1, :], I_01[:, 0:1, :]), 1)), I_01[:, 0:1, :]) + torch.nn.functional.mse_loss(self.model_Conv1D(torch.cat((I_02[:, 0:1, :], I_02[:, 0:1, :]), 1)), I_02[:, 0:1, :])
        loss_reg = torch.nn.functional.mse_loss(I_d[:, 0:1, :], 1/3 * (I_01[:, 0:1, :] + I_01[:, 3:, :])) + torch.nn.functional.mse_loss(I_1pd[:, 0:1, :], 1/3 * (I_02[:, 0:1, :] + I_02[:, 3:, :]))
        loss_reg_2 = torch.nn.functional.mse_loss(I_d2[:, 0:1, :], 2/3 * (I_01[:, 0:1, :] + I_01[:, 3:, :])) + torch.nn.functional.mse_loss(I_1pd2[:, 0:1, :], 2/3 * (I_02[:, 0:1, :] + I_02[:, 3:, :]))
        #loss = 0.25 * loss_1 + 0.05 * loss_1sd + 0.5 * loss_2 + 0.05 * loss_d + 0.05 * loss_2d + 0.05 * loss_1pd + 0.05 * loss_1p2d 
        #loss = 0.2 * loss_1 + 0.00 * loss_1sd + 0.00 * loss_2 + 0.20 * loss_d + 0.20 * loss_2d + 0.20 * loss_1pd + 0.20 * loss_1p2d 
        #loss = 0.5 * loss_1 + 0.00 * loss_1sd + 0.45 * loss_2 + 0.0125 * loss_d + 0.0125 * loss_2d + 0.0125 * loss_1pd + 0.0125 * loss_1p2d 
        loss = 0.5 * loss_1 + 0.00 * loss_1sd + 0.45 * loss_2 + 0.1 * loss_bound     #+ 0.025 * loss_reg + 0.025 * loss_reg_2
        #LOSS TRIALS
        #delta_1 = I_d[:, 0:1, :] - I_01[:, 0:1, :]
        #delta_2 = I_01[:, 3:, :] - 0.5 * (I_d[:, 1:2, :] + I_d2[:, 0:1, :])
        #delta_3 = 0.5 * (I_1pd[:, 0:1, :] + I_1pd2[:, 1:2, :]) - I_02[:, 0:1, :]
        #delta_4 = I_02[:, 3:, :] - I_1pd[:, 1:2, :]
        #loss_dyn = 0.5 * (torch.abs(delta_1 - delta_2) + torch.abs(delta_3 - delta_4)).mean()
        #loss = 0.5 * ( 0.45 * loss_1 + 0.45 * loss_2 + 0.1 * loss_bound + loss_dyn )
        #loss = loss_dyn
        #interp_1 = (2/3) * I_01[:, 0:1, :] + (1/3) * I_01[:, 3:, :]
        #interp_2 = (1/3) * I_01[:, 0:1, :] + (2/3) * I_01[:, 3:, :]
        #interp_4 = (2/3) * I_02[:, 0:1, :] + (1/3) * I_02[:, 3:, :]
        #interp_5 = (1/3) * I_02[:, 0:1, :] + (2/3) * I_02[:, 3:, :]
        #loss_interp = (torch.abs(I_d[:, 0:1, :] - interp_1).mean() +torch.abs(0.5*(I_d2[:, 0:1, :]+I_d[:, 1:2, :]) - interp_2).mean() + torch.abs(0.5*(I_1pd[:, 0:1, :] + I_1pd2[:, 1:2, :]) - interp_4).mean() + torch.abs(I_1pd[:, 1:2, :] - interp_5).mean())
        #loss = loss_dyn - 0.001 * loss_interp
        #LOSS H + d*W1
        #loss_H_1 = (torch.nn.functional.smooth_l1_loss(I_d2[:, 1:2, :], targets, beta=0.1) + torch.nn.functional.smooth_l1_loss(I_1pd2[:, 0:1, :], targets, beta=0.1)) / 2 
        #loss_H_2 = (torch.nn.functional.smooth_l1_loss(I_d[:, 1:2, :], I_d2[:, 0:1, :], beta=0.1) + torch.nn.functional.smooth_l1_loss(I_1pd[:, 0:1, :], I_1pd2[:, 1:2, :], beta=0.1)) / 2 
        #loss_H_bound = torch.nn.functional.smooth_l1_loss(self.model_Conv1D(torch.cat((I_01[:, 0:1, :], I_01[:, 0:1, :]), 1)), I_01[:, 0:1, :], beta=0.1) + torch.nn.functional.smooth_l1_loss(self.model_Conv1D(torch.cat((I_02[:, 0:1, :], I_02[:, 0:1, :]), 1)), I_02[:, 0:1, :], beta=0.1)
        #loss_EMD_1 = EMD_loss(I_d2[:, 1:2, :], targets) + EMD_loss(I_1pd2[:, 0:1, :], targets)
        #loss_EMD_2 = EMD_loss(I_d[:, 1:2, :], I_d2[:, 0:1, :]) + EMD_loss(I_1pd[:, 0:1, :], I_1pd2[:, 1:2, :])
        #loss_EMD_bound = EMD_loss(self.model_Conv1D(torch.cat((I_01[:, 0:1, :], I_01[:, 0:1, :]), 1)), I_01[:, 0:1, :]) + EMD_loss(self.model_Conv1D(torch.cat((I_02[:, 0:1, :], I_02[:, 0:1, :]), 1)), I_02[:, 0:1, :])
        #loss = 0.45 * (loss_H_1 + loss_EMD_1) + 0.45 * (loss_H_2 + loss_EMD_2) + 0.1 * (loss_H_bound + loss_EMD_bound).  #12/18_05
        #loss = 0.45 * (loss_H_1 + 0.1 *loss_EMD_1) + 0.45 * (loss_2) + 0.1 * (loss_H_bound + 0.1 * loss_EMD_bound)      #12/18_06, 12/23_00, 12/23_01
        #loss = 0.45 * (loss_1 + 0.1 *loss_EMD_1) + 0.45 * (loss_2) + 0.1 * (loss_bound + 0.1 * loss_EMD_bound)       #12/23_02
        #loss_H = (1/3) * (torch.nn.functional.smooth_l1_loss(I_01[:,0:1,:], I_d[:,0:1,:], beta=0.1) + torch.nn.functional.smooth_l1_loss(I_d[:, 1:2, :], I_d[:,0:1,:], beta=0.1) + 
        #                 torch.nn.functional.smooth_l1_loss(I_d[:, 1:2, :], targets, beta=0.1))
        #loss_EMD = (1/3) * (EMD_loss(I_01[:,0:1,:], I_d[:,0:1,:]) + EMD_loss(I_d[:, 1:2, :], I_d[:,0:1,:]) + EMD_loss(I_d[:, 1:2, :], targets))
        #loss = 0.5 * loss_m + 0.5 * (loss_H + 0.1 * loss_EMD).   #12/18_07
        #loss = 0.2 * loss_m + 0.8 * (loss_H + 0.1 * loss_EMD)     #12/18_08, 12/18_09
        #loss_H = (1/6) * (torch.nn.functional.smooth_l1_loss(I_01[:,0:1,:], I_d[:,0:1,:], beta=0.1) + torch.nn.functional.smooth_l1_loss(I_d[:, 0:1, :], I_d2[:,0:1,:], beta=0.1) + 
        #                 torch.nn.functional.smooth_l1_loss(I_d2[:, 0:1, :], targets, beta=0.1) + torch.nn.functional.smooth_l1_loss(I_02[:,0:1,:], I_1pd2[:,1:2,:], beta=0.1) + 
        #                 torch.nn.functional.smooth_l1_loss(I_1pd2[:, 1:2, :], I_1pd[:,1:2,:], beta=0.1) + torch.nn.functional.smooth_l1_loss(I_1pd[:, 1:2, :], I_02[:, 3:, :], beta=0.1))
        #loss_EMD = (1/6) * (EMD_loss(I_01[:,0:1,:], I_d[:,0:1,:]) + EMD_loss(I_d[:, 0:1, :], I_d2[:,0:1,:]) + EMD_loss(I_d2[:, 0:1, :], targets) + 
        #                    EMD_loss(I_02[:,0:1,:], I_1pd2[:,1:2,:]) + EMD_loss(I_1pd2[:, 1:2, :], I_1pd[:,1:2,:]) + EMD_loss(I_1pd[:, 1:2, :], I_02[:, 3:, :]))
        #loss = 0.25 * loss_m + 0.75 * (loss_H + 0.1 * loss_EMD)     #12/22_03
        #loss = 0.4 * loss_m + 0.6 * (loss_H + 0.2 * loss_EMD)       #12/22_04
        #loss = loss_m + 0.1 * loss_EMD                              #12/22_05
        #loss = loss_m + 0.4 * loss_EMD                               #12/22_06
        #x1_lin = (2/3) * I_01[:, 0:1, :] + (1/3) * I_01[:, 0:1, :]
        #x2_lin = (1/3) * I_01[:, 0:1, :] + (2/3) * I_01[:, 0:1, :]
        #x4_lin = (2/3) * I_02[:, 0:1, :] + (1/3) * I_02[:, 0:1, :]
        #x5_lin = (1/3) * I_02[:, 0:1, :] + (2/3) * I_02[:, 0:1, :]
        #loss_res = 0.25 * (torch.nn.functional.mse_loss(I_d[:, 0:1, :], x1_lin) + torch.nn.functional.mse_loss(0.5 * (I_d2[:, 0:1, :]+I_d[:, 1:2, :]), x2_lin) + torch.nn.functional.mse_loss(0.5 * (I_1pd[:, 0:1, :]+I_1pd2[:, 1:2,:]), x4_lin) + torch.nn.functional.mse_loss(I_1pd[:, 1:2, :], x5_lin))
        #loss = loss_m - 0.1 * loss_res        #12/22_07
        #loss = loss_m - 0.2 * loss_res        #12/22_08
        print("loss", loss)
        loss_over_GPUs = loss.detach().clone()
        loss.backward()
        losses.append(loss_over_GPUs.item())
        self.optimizer.step()

    def _run_epoch_f(self, epoch, batch_size, save_dir):
        b_sz = len(next(iter(self.train_data))[0])
        losses = []
        for data in self.train_data:
            source_sx, source_dx = self._data_train_preprocess_f(data, batch_size)
            list_sources = [source_sx.to(self.device), source_dx.to(self.device)]
            #n = len(data[-1]).  #questo era quello evcchio, prima di cambiare il dataloader per includere il batch size
            n = data[1].shape[1]
            l = n//2
            assert l == 3
            var = data[1][:, l:l+1, :].to(self.device)    #data[-1][l].to(self.device) 
            self._run_batch_f(list_sources, var, losses)  
        assert len(losses) == len(self.train_data)
        with open(save_dir + "/loss_file.txt", 'a') as f:    
            f.write(str([min(losses), np.percentile(losses, 25), np.percentile(losses, 50), np.percentile(losses, 75), max(losses)]) + '\n')
        print(f"losses mean epoch {epoch}", sum(losses)/len(losses))


    def _save_checkpoint_f(self, epoch, save_dir):
        ckp = self.model_Conv1D.state_dict()  
        save_dir_model = save_dir + "/model"
        if not os.path.exists(save_dir_model):
            os.makedirs(save_dir_model)
        PATH = save_dir_model + "/model_" + str(epoch) + ".pt"
        torch.save(ckp, PATH)

    def train_f(self, max_epochs: int, save_dir: str, batch_size: int, save_every: int, my_lr: float):
        for epoch in range(max_epochs + 1):
            print("epoch", epoch)
            self._run_epoch_f(epoch, batch_size, save_dir)
            if epoch % self.save_every == 0:
                self._save_checkpoint_f(epoch, save_dir)


def load_train_objs_f(path_ds_train, save_dir, my_lr):
    #normalization_stats = compute_and_save_normalization_stats_f(DataLoader(FloatDatasetFeat(path_ds_train, save_dir + "/normalization_stats.json")), save_dir)
    #print("end normalization stats", flush=True)
    train_set = FloatDatasetFeat(path_ds_train, save_dir + "/normalization_stats.json") 
    model = UNet1D()
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params=params, lr=my_lr)
    return train_set, model, optimizer


def prepare_dataloader_f(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=None
    )


def main_f(save_every: int, total_epochs: int, batch_size: int, path_ds_train: str, path_ds_test: str, save_dir: str, my_lr: float, alpha_smooth_reg: float, my_variable: str, my_epochs:int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("path ds train", path_ds_train)
    dataset, model, optimizer = load_train_objs_f(path_ds_train, save_dir, my_lr)
    train_data = prepare_dataloader_f(dataset, batch_size)
    trainer = TrainerFeat(model, train_data, optimizer, device, save_every, alpha_smooth_reg)
    trainer.train_f(total_epochs, save_dir, batch_size, save_every, my_lr)
    train_losses = read_losses_median_f(save_dir + "/loss_file.txt")
    plot_loss_f(train_losses, total_epochs, 1, save_dir + "/losses_plot") 

    plot_profiles_f(prepare_dataloader_f(dataset, batch_size=1),
                variable=my_variable,
                dir=save_dir,
                multiple_GPUs=False, 
                ep=my_epochs, train=True, mov_average=False)
    

    val_dataset = FloatDatasetFeat(path_ds_test, save_dir + "/normalization_stats.json") 
    print("path ds test", path_ds_test)
    plot_profiles_f(prepare_dataloader_f(val_dataset, batch_size=1),
                variable=my_variable,
                dir=save_dir,
                multiple_GPUs=False, 
                ep=my_epochs, train=False, mov_average=False)
    
    plot_NN_intermediate_profiles(prepare_dataloader_f(dataset, batch_size=1),
                variable=my_variable,
                dir=save_dir,
                ep=my_epochs, mov_average=False)
    
    linear_interpolation_comparison(prepare_dataloader_f(dataset, batch_size=1),
                variable=my_variable,
                dir=save_dir,
                ep=my_epochs, mov_average=False)
    
    RMSE_comparison(prepare_dataloader_f(dataset, batch_size=1),
                variable=my_variable,
                dir=save_dir,
                ep=my_epochs, mov_average=False)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate of Adam optimizer')
    parser.add_argument('--variable', type=str, default="CHLA", choices=["NITRATE", "CHLA", "BBP700", "TEMP", "DOXY"])
    parser.add_argument('--flag_toy', type=bool, default=False)
    parser.add_argument('--alpha_smooth_reg', default=0.0001, type=float, help='Profiles smoothing coefficient')
    args = parser.parse_args()

    #path_ds_train = f"../dataset/{args.variable}/toy_ds_sf_merged_train.csv" if args.flag_toy else f"../dataset/{args.variable}/complete_train_mnt.csv"
    #path_ds_test = f"../dataset/{args.variable}/toy_ds_sf_merged_test.csv" if args.flag_toy else f"../dataset/{args.variable}/complete_test_mnt.csv"
    path_ds_train = f"../dataset/{args.variable}/toy_ds_sf_train_mnt.csv" if args.flag_toy else f"../dataset/{args.variable}/complete_train_mnt.csv"
    path_ds_test = f"../dataset/{args.variable}/toy_ds_sf_test_mnt.csv" if args.flag_toy else f"../dataset/{args.variable}/complete_test_mnt.csv"
    #path_ds_train = f"../dataset/{args.variable}/toy_ds_sf_train_mnt.csv" if args.flag_toy else f"../dataset/{args.variable}/complete_train_mnt_reduced.csv"
    #path_ds_test = f"../dataset/{args.variable}/toy_ds_sf_test_mnt.csv" if args.flag_toy else f"../dataset/{args.variable}/complete_test_mnt_reduced.csv"
    save_dir = create_directories_f(["results_trial_1Unet", args.variable], date.today())
    save_ds_info_f(variable=args.variable,
                 flag_toy=args.flag_toy,
                 batch_size=args.batch_size,
                 epochs=args.total_epochs,
                 lr=args.lr,
                 save_dir=save_dir,
                 alpha_smooth_reg=args.alpha_smooth_reg)
    start_time = time.time()
    main_f(args.save_every, args.total_epochs, args.batch_size, path_ds_train, path_ds_test, save_dir, args.lr, args.alpha_smooth_reg, args.variable, args.total_epochs)
    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    print(f"[INFO] Total training time: {elapsed_minutes:.2f} minutes", flush=True)