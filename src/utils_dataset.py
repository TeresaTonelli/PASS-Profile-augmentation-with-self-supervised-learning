import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset

n_times = 7

def from_string_to_tensor_f(string, type):
    threshold = 99999
    label = 1 
    if type == "scalar":
        string = string.strip("[]").split(",")
    elif type == "tensor":
        string = string.replace("[", "").replace("]", "").split(",")
    len_single_start_data = len(string)
    out = torch.zeros(len_single_start_data)   #prima era 200, ora e` 7`  
    for ind in range(len(string)):
        if float(string[ind]) >= threshold:
            label = 0
        out[ind] = torch.tensor(float(string[ind]))
        #out[ind] = float(string[ind])
    return out, label


    ##string = string[8:-2].split(",")  #QUESTO VA BENE ER I TENSORI, dove li salvo con tensor()
    #string = string.strip("[]").split(",")    #Questo e' nuovo
    #print("string", string)
    #out = torch.zeros(len_single_start_data)   #prima era 200, ora e` 7`  
    #for ind in range(len(string)):
    #    if float(string[ind]) >= threshold:
    #        label = 0
    #    out[ind] = torch.tensor(float(string[ind]))
    #return out, label


#def compute_and_save_normalization_stats_f(loader, save_path):
#    all_days, all_years, all_lats, all_lons = [], [], [], []
#    all_temps, all_psals, all_doxys = [], [], []
#    for n_float, year, day, lat, lon, temp, psal, doxy, _ in loader:
#        all_days.append(day.flatten())
#        all_years.append(year.flatten())
#        all_lats.append(lat.flatten())
#        all_lons.append(lon.flatten())
#        all_temps.append(temp.flatten())
#        all_psals.append(psal.flatten())
#        all_doxys.append(doxy.flatten())
#    def stack_and_stats_f(tensors):
#        cat = torch.cat(tensors)
#        return float(cat.mean()), float(cat.std())
#    stats = {
#        "day": stack_and_stats_f(all_days),
#        "year": stack_and_stats_f(all_years),
#        "lat": stack_and_stats_f(all_lats),
#        "lon": stack_and_stats_f(all_lons),
#        "temp": stack_and_stats_f(all_temps),
#        "psal": stack_and_stats_f(all_psals),
#        "doxy": stack_and_stats_f(all_doxys)
#    }
#    with open(os.path.join(save_path, "normalization_stats.json"), "w") as f:
#        json.dump(stats, f, indent=4)
#    print(f"[INFO] Saved normalization stats to {os.path.join(save_path, 'normalization_stats.json')}", flush=True)
#    return stats


def compute_and_save_normalization_stats_f(loader, save_path):
    all_days, all_years, all_lats, all_lons = [], [], [], []  #torch.zeros([200, 7]), torch.zeros([200, 7]), torch.zeros([200, 7]), torch.zeros([200, 7])
    all_temps, all_psals, all_doxys = [], [], []    #torch.zeros([200, 7]), torch.zeros([200, 7]), torch.zeros([200, 7])
    for n_float, data_full in loader:
        years_ntuple = data_full[:, :, :, 0, :]
        day_rad_ntuple = data_full[:, :, :, 1, :]
        #print("day rad ntuple", day_rad_ntuple)
        #print("day n tuples", day_rad_ntuple.shape)
        lat_ntuple = data_full[:, :, :, 2, :]
        lon_ntuple = data_full[:, :, :, 3, :]
        temp_ntuple = data_full[:, :, :, 4, :]
        #print("temp full n tuples", temp_full.shape)
        #print("temp full tensor 1D", temp_full[:, :10, :, 0:].squeeze())
        psal_ntuple = data_full[:, :, :, 5, :]
        doxy_ntuple = data_full[:, :, :, 6, :]
        var_ntuple = data_full[:, :, :, -1, :]
        print("day squeeze", day_rad_ntuple.squeeze().shape)
        #all_days = torch.cat((all_days, day_rad_ntuple.squeeze()), axis=1)
        #all_years = torch.cat((all_years, years_ntuple.squeeze()), axis=1)
        #all_lats = torch.cat((all_lats, lat_ntuple.squeeze()), axis=1)
        #all_lons = torch.cat((all_lons, lon_ntuple.squeeze()), axis=1)
        #all_temps = torch.cat((all_temps, temp_ntuple.squeeze()), axis=1)
        #all_psals = torch.cat((all_psals, psal_ntuple.squeeze()), axis=1)
        #all_doxys = torch.cat((all_doxys, doxy_ntuple.squeeze()), axis=1)
        all_days.append(day_rad_ntuple)
        all_years.append(years_ntuple)
        all_lats.append(lat_ntuple)
        all_lons.append(lon_ntuple)
        all_temps.append(temp_ntuple)
        all_psals.append(psal_ntuple)
        all_doxys.append(doxy_ntuple)
    #print("all day shape", all_days.shape)
    #all_days = all_days[:, :7]
    def stack_and_stats_f(tensors):
        cat = torch.cat(tensors)
        print("cat shape", cat.shape)
        return float(cat.mean()), float(cat.std())
    stats = {
        "day": stack_and_stats_f(all_days),
        "year": stack_and_stats_f(all_years),
        "lat": stack_and_stats_f(all_lats),
        "lon": stack_and_stats_f(all_lons),
        "temp": stack_and_stats_f(all_temps),
        "psal": stack_and_stats_f(all_psals),
        "doxy": stack_and_stats_f(all_doxys)
    }
    with open(os.path.join(save_path, "normalization_stats.json"), "w") as f:
        json.dump(stats, f, indent=4)
    print(f"[INFO] Saved normalization stats to {os.path.join(save_path, 'normalization_stats.json')}", flush=True)
    return stats


class FloatDatasetFeat(Dataset):

    def __init__(self, path_df=None, stats_path="normalization_stats.json"):
        super().__init__()
        if path_df is not None:
            self.path_df = path_df
            self.df = pd.read_csv(self.path_df)
        else:
            raise Exception("Paths should be given as input to initialize the Float class.")

        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                self.norm_stats = json.load(f)
        else:
            self.norm_stats = None  

    def normalize(self, x, name):
        if self.norm_stats is not None and name in self.norm_stats:
            print("perform normalization in dataloader")
            mean, std = self.norm_stats[name]
            return (x - mean) / std
        return x

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.df.iloc[0, :])

    def __getitem__(self, index):
        """Generates one sample of data"""
        try:
            self.samples = self.df.iloc[:, index + 1].tolist() 
            self.column_name = self.df.columns[index + 1]
        except Exception as error:
            pass

        year_full, label_year_full = from_string_to_tensor_f(self.samples[0], "scalar")
        day_rad_full, label_day_rad_full = from_string_to_tensor_f(self.samples[1], "scalar")
        lat_full, label_lat_full = from_string_to_tensor_f(self.samples[2], "scalar")
        lon_full, label_lon_full = from_string_to_tensor_f(self.samples[3], "scalar")
        year_list_n = [self.normalize(year_, "year") for year_ in year_full]
        day_rad_list_n = [self.normalize(day_rad_, "year") for day_rad_ in day_rad_full]
        lat_list_n = [self.normalize(lat_, "year") for lat_ in lat_full]
        lon_list_n = [self.normalize(lon_, "year") for lon_ in lon_full]

        temp_full, label_temp_full = from_string_to_tensor_f(self.samples[4], "tensor")
        psal_full, label_psal_full = from_string_to_tensor_f(self.samples[5], "tensor")
        doxy_full, label_doxy_full = from_string_to_tensor_f(self.samples[6], "tensor")
        variable_full, label_variable_full = from_string_to_tensor_f(self.samples[7], "tensor")    

        temp_list = []
        psal_list = []
        doxy_list = []
        variable_list = []
        for j in range(n_times):
            temp_ = torch.tensor([temp_full[i] for i in range(1400) if i%n_times==j])
            temp_list.append(temp_)
            psal_ = torch.tensor([psal_full[i] for i in range(1400) if i%n_times==j])
            psal_list.append(psal_)
            doxy_ = torch.tensor([doxy_full[i] for i in range(1400) if i%n_times==j])
            doxy_list.append(doxy_)
            variable_ = torch.tensor([variable_full[i] for i in range(1400) if i%n_times==j])
            variable_list.append(variable_)

        temp_list_n = [self.normalize(temp_, "temp") for temp_ in temp_list]   #in questo modo io normalizzo un profilo alla volta, separatamente    
        psal_list_n = [self.normalize(psal_, "psal") for psal_ in psal_list]
        doxy_list_n = [self.normalize(doxy_, "doxy") for doxy_ in doxy_list]

        label = label_doxy_full * label_psal_full * label_temp_full  

        #return self.column_name, year_list_n, day_rad_list_n, lat_list_n, lon_list_n, temp_list_n, psal_list_n, doxy_list_n, variable_list

        #Addition for batch
        year_tensor = torch.stack([year.repeat(200) for year in year_list_n])
        day_tensor = torch.stack([day.repeat(200) for day in day_rad_list_n])
        lat_tensor = torch.stack([lat.repeat(200) for lat in lat_list_n])
        lon_tensor = torch.stack([lon.repeat(200) for lon in lon_list_n])
        temp_tensor = torch.stack([temp for temp in temp_list_n])
        psal_tensor = torch.stack([psal for psal in psal_list_n])
        doxy_tensor = torch.stack([doxy for doxy in doxy_list_n])
        var_tensor = torch.stack([var for var in variable_list])
        assert year_tensor.shape == lat_tensor.shape == temp_tensor.shape == psal_tensor.shape == doxy_tensor.shape == var_tensor.shape == torch.Size([7, 200])
        
        # Stack all features as channels → [C=4, D=n_times, H=depth]
        input_tensor = torch.stack([year_tensor, day_tensor, lat_tensor, lon_tensor, temp_tensor,psal_tensor, doxy_tensor,var_tensor])
        assert input_tensor.shape == torch.Size([8, 7, 200]) # Shape: [4, n_times, depth]

        # Expand to [C, D, H, W] for Conv3D (W=1 dummy width)
        input_tensor = input_tensor.unsqueeze(-1)  # → [4, n_times, depth, 1]
        input_tensor = input_tensor.permute(3, 2, 0, 1)



        return self.column_name, input_tensor

        #return self.column_name, year_full, day_rad_full, lat_full, lon_full, temp, psal, doxy, variable
        #return self.column_name, year_list_n, day_rad_list_n, lat_list_n, lon_list_n, temp_list_n, psal_list_n, doxy_list_n, variable_list