from typing import List

import numpy as np
import torch

class DisplayParams:
    """
    Displays all the learnable parameters in a tabular format.
    Columns of this table:
    Layer Name: which layer it belongs to in the model architecture.
    Type of Parameter: whether it is conv, batchnorm, linear, etc.
    Shape of the learnable paramteres: 4d np array
    Total leanrable parameter: int

    Args:
        layer_name: List containing names of layers
        layer_params_shape: 4d Tensor.
        learn_params: total learnable params calculated for that layer.
    """
    def __init__(self, ):
        super(DisplayParams, self).__init__()

    def get_memory_per_params(layer_name: List[str, ],
                              learn_params_shape: np.array,
                              learn_params: int,
        ):
        print(type(layer_name[0]))
        print("\n----------------------------------------------------------------------------------------------------------------------------------")
        line_new = "{:>25} {:>30} {:>35} {:>30}".format(
                "Layer Name",
                "Param Type",
                "Shape of Learnable Parameters",
                "Total Learnable Parameters",
                )
        print(line_new)
        print("===================================================================================================================================")
        for i, each_layer in enumerate(layer_name):
            full_layer = each_layer.split('.')
            param_type = '.'.join(full_layer[-2:])
            each_layer_name = '.'.join(full_layer[:-2])
            line_new = "{:>25} {:>30} {:>35} {:>30}".format(
                    each_layer_name,
                    param_type,
                    str(list(learn_params_shape[i])),
                    learn_params[i]
                )
            print(line_new)
        print("====================================================================================================================================")
        print()
