import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
from datetime import date
import time
import numpy as np

from NN_architecture import DoubleConv, UNet3D
from plots import plot_loss_f, plot_profiles_f
from utils_dataset import FloatDatasetFeat, compute_and_save_normalization_stats_f
from utils_plots import read_losses_median_f
from utils_src import create_directories_f, save_ds_info_f

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
        #training_year_ntuple = [input_data[1][j].unsqueeze(1).to(self.device) for j in range(len(input_data[1]))]
        #training_day_ntuple = [input_data[2][j].unsqueeze(1).to(self.device) for j in range(len(input_data[2]))]
        #training_lat_ntuple = [input_data[3][j].unsqueeze(1).to(self.device) for j in range(len(input_data[3]))]
        #training_lon_ntuple = [input_data[4][j].unsqueeze(1).to(self.device) for j in range(len(input_data[4]))]
        #training_temp_ntuple = [input_data[5][j].to(self.device) for j in range(len(input_data[5]))]
        #training_psal_ntuple = [input_data[6][j].to(self.device) for j in range(len(input_data[6]))]
        #training_doxy_ntuple = [input_data[7][j].to(self.device) for j in range(len(input_data[7]))]
        #training_var_ntuple = [input_data[-1][j].to(self.device) for j in range(len(input_data[-1]))]
        #n = len(training_year_ntuple)
        #v = len(input_data) - 1
        #assert v == 8
        #assert n == 7
        #assert len(training_year_ntuple) == len(training_day_ntuple) == len(training_lat_ntuple) == len(training_lon_ntuple)
        #assert len(training_year_ntuple) == len(training_temp_ntuple) == len(training_psal_ntuple) ==len(training_doxy_ntuple) == len(training_var_ntuple)
        #l = n//2 
        #d = training_temp_ntuple[0].shape[1]
        #assert d == 200
        ##Firstly I concatenate all the covariates --> and I repeat the values of the pointwise measures
        #training_x = torch.zeros([d, v, n])
        #for i_n in range(n):
        #    training_x_in = torch.cat((training_year_ntuple[i_n].repeat(d, 1), training_day_ntuple[i_n].repeat(d, 1), training_lat_ntuple[i_n].repeat(d, 1), 
        #                               training_lon_ntuple[i_n].repeat(d, 1), training_temp_ntuple[i_n].t(), training_psal_ntuple[i_n].t(), 
        #                               training_doxy_ntuple[i_n].t(), training_var_ntuple[i_n].t()), dim=1)
        #    training_x[:, :, i_n] = training_x_in
        n = training_x.shape[-1]
        l = n//2
        ##Split between training sx and training dx
        training_x_sx = training_x[:, :, :, :, 0:l+1] 
        training_x_dx = training_x[:, :, :, :, l:n+1]
        assert (training_x_sx[:, :, :, :, 3] == training_x_dx[:, :, :, :, 0]).all()
        #print("training x sx shape", training_x_sx.shape)
        #assert training_x_sx.shape == torch.Size([batch_size, 1, 200, 8, l+1]) 
        return training_x_sx, training_x_dx

        #training_x_tmp = torch.cat((training_year, training_day, training_lat, training_lon), dim=0)
        #assert training_x_tmp.shape  == torch.Size([4, 1])
        #training_x_single = training_x_tmp.clone()
        #for j in range(n-1):
        #    training_x_tmp = torch.cat((training_x_tmp, training_x_single), dim=1)
        #assert training_x_tmp.shape == torch.Size([4, n])
        #assert (training_x_tmp[:, 2] == training_x_tmp[:, 6]).all()
        ##Add th tensor variables
        #training_x_tmp = torch.cat((training_x_tmp, training_temp, training_psal, training_doxy, training_var), dim=0)
        #assert training_x_tmp.shape == torch.Size([8, n])
        #training_x_sx = training_x_tmp[:, 0:l+1] 
        #training_x_dx = training_x_tmp[:, l:n+1] 
        #assert (training_x_sx[:, 3] == training_x_dx[:, 0]).all()
        #assert training_x_sx.shape == torch.Size([8, l+1]) 
        #return training_x_sx, training_x_dx

    def _run_batch_f(self, list_sources, targets, losses):
        self.optimizer.zero_grad()
        I_01, I_02 = list_sources
        #print("targets", targets)
        assert (targets == I_01[:, :, :, -1:, -1:]).all()
        assert (targets == I_02[:, :, :, -1:, 0:1]).all()
        #print("input shape", I_01[:, :, :, :, [0,3]].shape) 
        I_d = self.model_Conv1D(I_01[:, :, :, :, [0,3]])  
        #print("Id output shape", I_d.shape)  
        I_1pd = self.model_Conv1D(I_02[:, :, :, :, [0,3]])
        #print("ID2 cat shpae", torch.cat((I_d[:, :, :, :, 0:1], I_1pd[:, :, :, :, 0:1]), -1).shape)
        I_d2 = self.model_Conv1D(torch.cat((I_d[:, :, :, :, 0:1], I_1pd[:, :, :, :, 0:1]), -1))    
        I_1pd2 = self.model_Conv1D(torch.cat((I_d[:, :, :, :, 1:2], I_1pd[:, :, :, :, 1:2]), -1))     
        loss_1 = (torch.nn.functional.mse_loss(I_d2[:, :, :, -1:, 1:2], targets) + torch.nn.functional.mse_loss(I_1pd2[:, :, :, -1:, 0:1], targets)) / 2      
        loss_1sd = torch.nn.functional.mse_loss(I_d2[:, :, :, -1:, 1:2], I_1pd2[:, :, :, -1:, 0:1])     
        loss_2 = (torch.nn.functional.mse_loss(I_d[:, :, :, -1:, 1:2], I_d2[:, :, :, -1:, 0:1]) + torch.nn.functional.mse_loss(I_1pd[:, :, :, -1:, 0:1], I_1pd2[:, :, :, -1:, 1:2])) / 2      
        loss_d = torch.nn.functional.mse_loss(I_d[:, :, :, -1:, 0:1], I_01[:, :, :, -1:, 1:2])     
        loss_2d = (torch.nn.functional.mse_loss(I_d[:, :, :, -1:, 1:2], I_01[:, :, :, -1:, 2:3]) + torch.nn.functional.mse_loss(I_d2[:, :, :, -1:, 0:1], I_01[:, :, :, -1:, 2:3])) / 2       
        loss_1pd = (torch.nn.functional.mse_loss(I_1pd[:, :, :, -1:, 0:1], I_02[:, :, :, -1:, 1:2]) + torch.nn.functional.mse_loss(I_1pd2[:, :, :, -1:, 1:2], I_02[:, :, :, -1:, 1:2])) / 2   
        loss_1p2d = torch.nn.functional.mse_loss(I_1pd[:, :, :, -1:, 1:2], I_02[:, :, :, -1:, 2:3])   
        loss = 0.25 * loss_1 + 0.05 * loss_1sd + 0.5 * loss_2 + 0.05 * loss_d + 0.05 * loss_2d + 0.05 * loss_1pd + 0.05 * loss_1p2d 
        #print("loss", loss)
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
            n = data[1].shape[-1]
            l = n//2
            assert l == 3
            var = data[1][:, :, :, -1:, l:l+1].to(self.device)    #data[-1][l].to(self.device) 
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
    #print("end normalization stats")
    train_set = FloatDatasetFeat(path_ds_train, save_dir + "/normalization_stats.json") 
    model = UNet3D()
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
                ep=my_epochs, train=True, mov_average=True)

    val_dataset = FloatDatasetFeat(path_ds_test, save_dir + "/normalization_stats.json") 
    print("path ds test", path_ds_test)
    plot_profiles_f(prepare_dataloader_f(val_dataset, batch_size=1),
                variable=my_variable,
                dir=save_dir,
                multiple_GPUs=False, 
                ep=my_epochs, train=False, mov_average=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate of Adam optimizer')
    parser.add_argument('--variable', type=str, default="CHLA", choices=["NITRATE", "CHLA", "BBP700"])
    parser.add_argument('--flag_toy', type=bool, default=False)
    parser.add_argument('--alpha_smooth_reg', default=0.0001, type=float, help='Profiles smoothing coefficient')
    args = parser.parse_args()

    path_ds_train = f"../dataset/{args.variable}/toy_merged_n_tuple.csv" if args.flag_toy else os.getcwd() + f"../dataset/{args.variable}/float_ds_sf_train_full_cp_no_out_strict_20.csv"
    path_ds_test = f"../dataset/{args.variable}/toy_merged_n_tuple.csv" if args.flag_toy else os.getcwd() + f"../dataset/{args.variable}/float_ds_sf_test_full.csv"
    save_dir = create_directories_f(["results_trial", args.variable], date.today())
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