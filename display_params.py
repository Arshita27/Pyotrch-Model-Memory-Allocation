import numpy
import torch


def display_params(layer_name, learn_params_shape, learn_params):
    # print("\n-------------------------------------------------------------------------------------------------------------")
    # line_new = "{:>45}  {:>30} {:>30}".format(
    #         "Layer Name",
    #         "Shape of Learnable Parameters",
    #         "Total Learnable Parameters")
    # print(line_new)
    # print("=============================================================================================================")
    # for i, each_layer in enumerate(layer_name):
    #     line_new = "{:>45}  {:>30} {:>30}".format(
    #             each_layer,
    #             str(list(learn_params_shape[i])),
    #             learn_params[i]
    #         )
    #     print(line_new)
    # print("=============================================================================================================")
    # print()


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
        # print(param_type, each_layer_name)
        line_new = "{:>25} {:>30} {:>35} {:>30}".format(
                each_layer_name,
                param_type,
                str(list(learn_params_shape[i])),
                learn_params[i]
            )
        print(line_new)
    print("====================================================================================================================================")
    print()




    # total_params = 0
    # total_output = 0
    # trainable_params = 0
    # for layer in summary:
    #     # input_shape, output_shape, trainable, nb_params
    #     line_new = "{:>20}  {:>25} {:>15}".format(
    #         layer,
    #         str(summary[layer]["output_shape"]),
    #         "{0:,}".format(summary[layer]["nb_params"]),
    #     )
    #     total_params += summary[layer]["nb_params"]
    #     total_output += np.prod(summary[layer]["output_shape"])
    #     if "trainable" in summary[layer]:
    #         if summary[layer]["trainable"] == True:
    #             trainable_params += summary[layer]["nb_params"]
    #     print(line_new)
    #
    # # assume 4 bytes/number (float on cuda).
    # total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    # total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    # total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    # total_size = total_params_size + total_output_size + total_input_size
    #
    # print("================================================================")
    # print("Total params: {0:,}".format(total_params))
    # print("Trainable params: {0:,}".format(trainable_params))
    # print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    # print("----------------------------------------------------------------")
    # print("Input size (MB): %0.2f" % total_input_size)
    # print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    # print("Params size (MB): %0.2f" % total_params_size)
    # print("Estimated Total Size (MB): %0.2f" % total_size)
    # print("----------------------------------------------------------------")
