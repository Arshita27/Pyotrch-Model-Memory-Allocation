import numpy as np
from psutil import virtual_memory
from pynvml import *
import torch
import torch.nn as nn
from torchvision import models


class MemoryAllocation():
    """
    ToDo: Write a short description.

    Args:
        model: Pytorch Model
        input: 4d Tensor, usually output of dataloader.
    """

    def __init__(
        self,
        model: nn.Module,
        input: torch.Tensor,
        device: str,
    ):
        self.model = model
        self.input_type = input.type()
        self.device = device

    def get_num_params(self, ) -> int:
        """
        Calculates the total number of parameters that need to be trained in the model.
        """
        total_params = 0
        for param_tensor in self.model.state_dict():
            # print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())
            params_layer = 1
            for each_dim in self.model.state_dict()[param_tensor].size():
                params_layer = params_layer * each_dim
            total_params += params_layer
        return total_params

    def get_memory_params(self, total_params, ) -> int:
        """
        Multiplies the total nnumber of parameters in the model with the type of input.
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
        print("Total memory required (including model parameters, input_type and batch_size):", memory_needed, "bits.")

        # if torch.cuda.is_available():
        #     nvmlInit()
        #     handle = nvmlDeviceGetHandleByIndex(0)
        #     info = nvmlDeviceGetMemoryInfo(handle)
        #     total_device_memory = info.total
        #     print("GPU Memory available: ", info.total, "bytes")
        if self.device == "GPU":
            while torch.cuda.is_available():
                try:
                    nvmlInit()
                    handle = nvmlDeviceGetHandleByIndex(0)
                    info = nvmlDeviceGetMemoryInfo(handle)
                    total_device_memory = info.total
                    print("GPU Memory available: ", info.total, "bytes")
                except:
                    print("GPU not found!")
        else:
            mem = virtual_memory()
            total_device_memory = mem.total
            print("CPU Memory available: ", total_device_memory, "bytes")
        used = ((memory_needed/8)/total_device_memory)*100
        print("Percentage of memory used after loading the given model:", '%.2f' % used, "%")
