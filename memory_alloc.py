import numpy as np
from psutil import virtual_memory
import torch
import torch.nn as nn
from torchvision import models
from pynvml import *


def get_num_params(model: nn.Module) -> int:
    total_params = 0
    for param_tensor in model.state_dict():
        # print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        params_layer = 1
        for each_dim in model.state_dict()[param_tensor].size():
            params_layer = params_layer * each_dim
        # print("Parameters per layer: ", params_layer)
        total_params += params_layer
    # print("Total parameters in the given model: ", total_params)
    return total_params

def get_memory_params(total_params: int, input: torch.Tensor) -> int:
    if input.type() == "torch.FloatTensor":
        return total_params*32
    elif input.type() == "torch.DoubleTensor":
        return total_params*64
    else:
        return Exception("Cannot understand input type.")

def get_memory_info(model: nn.Module, input: torch.Tensor):
    total_params = get_num_params(model)
    memory_needed = get_memory_params(total_params, input)
    print("Total params in the given model: ", total_params)
    print("Total memory required: ", memory_needed)

    if torch.cuda.is_available():
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        total_device_memory = info.total
        print("GPU Memory available: ", info.total)
    else:
        mem = virtual_memory()
        total_device_memory = mem.total
        print("CPU Memory available: ", total_device_memory)
    used = (memory_needed/total_device_memory)*100
    print("Percentage of memory used after loading the given model: ", used)


if __name__ == "__main__":
    img = np.random.random((572, 572, 3))
    input1 = torch.from_numpy(img).permute(2,0,1).type(torch.FloatTensor)
    input1 = input1.unsqueeze(0)
    input2 = input1.clone()
    input = torch.cat((input1, input2), dim=0)
    print("Shape of input: ", input.shape)

    model = models.resnet50()
    # print(model)
    get_memory_info(model, input)
