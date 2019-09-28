import numpy as np
from psutil import virtual_memory
import torch
import torch.nn as nn
from torchvision import models
from pynvml import *


class MemoryAllocation():
    """
    ToDo: Write a short description.

    Args:
        model: Pytorch Model
               (default = resnet18)
        input: 4d Tensor, usually output of dataloader.
               (default = (10, 3, 572, 572))
    """

    def __init__(
        self,
        model: nn.Module,
        input: torch.Tensor,
    ):
        self.model = model
        self.input_type = input.type()
        self.batch_size = input.shape[0]

    def get_num_params(self, ) -> int:
        """
        Calculates the total number of parameters that need to be trained in the model.
        """
        total_params = 0
        for param_tensor in self.model.state_dict():
            params_layer = 1
            for each_dim in self.model.state_dict()[param_tensor].size():
                params_layer = params_layer * each_dim
            total_params += params_layer
        return total_params

    def get_memory_params(self, total_params, ) -> int:
        """
        Multiplies the total nnumber of parameters in the model with the batch_size and type of input.
        The 'input_type' indicates whether the input is 4, 8, 16, 32 bits.
        """
        if self.input_type == "torch.FloatTensor":
            return total_params*32
        elif self.input_type == "torch.DoubleTensor":
            return total_params*64
        else:
            return Exception("Cannot understand input type.")

    def get_memory_info(self):
        """
        Get info of GPU/CPU memory and compare with the memory used to lead the model with given input.
        """
        total_params = self.get_num_params()
        memory_needed = self.get_memory_params(total_params)
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
