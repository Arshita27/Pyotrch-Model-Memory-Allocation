import numpy as np
import torch
from torchvision import models

from memory_alloc import MemoryAllocation


if __name__ == "__main__":

    img = np.random.random((3, 572, 572))
    input = torch.from_numpy(img).type(torch.FloatTensor)
    input = input.unsqueeze(0).expand(8, 3, 572, 572)
    print("Shape of input: ", input.shape)

    model = models.resnet18()

    MemoryAllocation(model,
                    input,
                    device="CPU",
                    give_summary=True).get_memory_info()
