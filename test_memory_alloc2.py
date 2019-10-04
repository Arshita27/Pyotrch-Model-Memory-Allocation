import numpy as np
import torch
from torchvision import models
import torch.nn as nn

from memory_alloc import MemoryAllocation


if __name__ == "__main__":

    img = np.random.random((3, 572, 572))
    input = torch.from_numpy(img).type(torch.FloatTensor)
    input = input.unsqueeze(0).expand(8, 3, 572, 572)
    print("Shape of input: ", input.shape)

    model = models.resnet18()
    MemoryAllocation(model, input, device="CPU").get_memory_info()

    input_channels = 3
    n = 2

    for module in model.modules():
        if (isinstance(module, nn.Conv2d)):
            model.out_channels = module.out_channels//n

            if module.in_channels != input_channels:
                module.in_channels = module.in_channels//n

        if (isinstance(module, nn.BatchNorm2d)):
            model.num_features = module.num_features//n

        if (isinstance(module, nn.Linear)):
            model.out_features = module.out_features//n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print("*************************After modifications***********************")
    MemoryAllocation(model, input, device="CPU").get_memory_info()
