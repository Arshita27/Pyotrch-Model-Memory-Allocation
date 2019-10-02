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

    def get_memory_params(self, total_params: int, ) -> int:
        """
        Multiplies the total nnumber of parameters in the model with the type of input.
        The 'input_type' indicates whether the input is 4, 8, 16, 32 bits.
        """
        if self.input_type == "torch.FloatTensor" or self.input_type == "torch.cuda.FloatTensor" :
            return total_params*32
        elif self.input_type == "torch.DoubleTensor" or self.input_type == "torch.cuda.DoubleTensor":
            return total_params*64
        elif self.input_type  == "torch.ByteTensor" or self.input_type == "torch.cuda.ByteTensor":
            return total_params*8
        else:
            return Exception("Cannot understand input type.")

    def gpu_get_memory(self, ):
        """
        Calculates memory in GPU.
        """
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        return info.total

    def cpu_get_memory(self, ):
        """
        Calculates memory in CPU.
        """
        mem = virtual_memory()
        return mem.total

    def get_memory_info(self):
        """
        Get info of GPU/CPU memory and compare with the memory used to lead the model with given input.
        """
        total_params = self.get_num_params()
        memory_needed = self.get_memory_params(total_params)
        print("Total params in the given model: ", total_params)
        print("Total memory required to load a total of {} model parameters and of input type {} is {} bits".format(
                                                                                total_params,
                                                                                self.input_type,
                                                                                memory_needed))
        if self.device == "GPU":
            if torch.cuda.is_available():
                total_device_memory = self.gpu_get_memory()
                print("GPU memory avaiable is {} bytes.".format(total_device_memory))

            else:
                print("GPU not found! Memory will be computed on CPU.")
                total_device_memory = self.cpu_get_memory()
                print("CPU memory avaiable is {} bytes.".format(total_device_memory))

        elif self.device == "CPU":
            total_device_memory = self.cpu_get_memory()
            print("CPU memory avaiable is {} bytes.".format(total_device_memory))

        else:
            raise Exception("'{}' device is not defined. Use one of the following: 'GPU' or 'CPU'.".format(self.device))

        used = ((memory_needed/8)/total_device_memory)*100
        print("Percentage of {} memory used after loading the given model is {}%".format(self.device, '%.2f' % used))
