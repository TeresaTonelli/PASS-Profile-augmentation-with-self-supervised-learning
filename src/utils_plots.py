import os
import torch 


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


def from_loader_data_to_train_data_f(input_data):
    testing_x = input_data[1]
    n = testing_x.shape[-1]
    l = n//2
    ##Split between training sx and training dx
    testing_x_sx = testing_x[:, 0:l+1, :] 
    testing_x_dx = testing_x[:, l:n+1, :]
    assert (testing_x_sx[:, 3, :] == testing_x_dx[:, 0, :]).all()
    return testing_x_sx, testing_x_dx