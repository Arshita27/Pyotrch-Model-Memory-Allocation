# Pyotrch-Model-Memory-Allocation

The Memory Allocation module has been created to let the users understand the number of parameters present in any pytorch model and whether such models can fit in the given memory on your device and how much of memory they take up.

The parameters in the model are converted into bits considering the input type that is fed into the model. (for eg: input type = FloatTensor corresponds to the fact that each paramter will be of size Float and will take 32 bit memory)

#### Packages that need to be installed:

  1. python: 3.6
  2. pytorch: 0.4 or higher
  3. torchvision: 0.2.1 (in case you are loading in torchvision model)
  4. psutil: pip install psutil
  5. For 'pynvm', you can use either of the two commands:

  pypi:
  >  pip3 install nvidia-ml-py3

  Conda:
  >  conda install nvidia-ml-py3 -c fastai

#### Information regarding arguments to the module.

The module takes three arguments as inputs.
  1. model: nn.Module (eg: resnet50 or any other pytorch model)
  2. input: torch.Tensor (eg: torch.Tensor)
  3. device: str ("GPU" or "CPU")

Note:
  * The module works irrespective of the fact that the model is loaded on GPU or not.
  * The module works irrespective of the fact that the input is loaded on GPU or not. It currently takes three input types:
    - torch.FloatTensor or torch.cuda.FloatTensor
    - torch.DoubleTensor or torch.cuda.DoubleTensor
    - torch.ByteTensor or torch.cuda.ByteTensor
    (more types will be added soon.)
  * If device = "GPU" is specified but not found while running this module, it will throw in a message stating that no "GPU was found" and will caluclate the memory allocation with respect to "CPU".


## __Note__:
If device == "GPU" is selected, the current module assumes that the entire pytorch model will stay on one GPU. There is no  current implementation for multiGPU use.
