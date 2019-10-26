import numpy as np
import torch
from torchvision import models

import PytorchModelMemoryMap as pm3


if __name__ == "__main__":

    # created random image
    img = np.random.random((3, 572, 572))
    input = torch.from_numpy(img).type(torch.FloatTensor)
    input = input.unsqueeze(0).expand(8, 3, 572, 572)

    # torchvision resnet model
    model = models.resnet18()

    # Call Memory Map Module
    pm3.MemoryMap(model,
            input,
            device="CPU",
            get_summary=True).get_memory_info()
