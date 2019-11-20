import numpy as np
import torch
from torch import nn

from core.layers import get_layers


class Darknet(nn.Module):
    def __init__(self, net_params, layer_infos):
        super(Darknet, self).__init__()
        self.layer_infos = layer_infos
        self.module_list = get_layers(net_params, layer_infos)

    def forward(self, x):
        detections = None
        # Cache outputs for future route & shortcut layers
        layer_outputs = []

        for layer_idx, (layer_info, layer) in \
                enumerate(zip(self.layer_infos, self.module_list)):
            layer_type = layer_info["type"]

            if layer_type in ["convolutional", "maxpool", "upsample"]:
                x = layer(x)

            elif layer_type == 'shortcut':
                prev_layer_idx = layer_info["from"]
                try:
                    x += layer_outputs[prev_layer_idx]
                except RuntimeError:
                    print(f"Shape not matching in shortcut layer, \
                          trying to add {layer_outputs[prev_layer_idx].shape} \
                          to {x.shape}")

            elif layer_type == "route":
                prev_layer_idxs = layer_info["layers"]
                if len(prev_layer_idxs) == 1:
                    x = layer_outputs[prev_layer_idxs[0]]
                else:
                    prev_layer_outputs = [layer_outputs[i]
                                          for i in prev_layer_idxs]
                    try:
                        x = torch.cat(prev_layer_outputs, dim=1)
                    except RuntimeError:
                        print("Spatial size not matching in route layers",
                              [out.shape for out in prev_layer_outputs])

            elif layer_type == "yolo":
                x = layer(x)
                if detections is None:
                    detections = x
                else:
                    detections = torch.cat((detections, x), dim=1)

            layer_outputs.append(x)
        return detections

    def load_weights(self, weight_path):
        with open(weight_path) as f:
            header = np.fromfile(f, dtype=np.int32, count=5)
            weights = np.fromfile(f, dtype=np.float32)
            print(len(weights))

        ptr = 0
        for i, layer in enumerate(self.module_list):
            layer_type = self.layer_infos[i]["type"]
            if layer_type == "convolutional":
                conv = layer[0]
                if "batch_normalize" in self.layer_infos[i]:
                    bn = layer[-1]
                    # Copy the pretrained weights to model params
                    ptr = self.__load_params(bn.bias, weights, ptr)
                    ptr = self.__load_params(bn.weight, weights, ptr)
                    ptr = self.__load_params(bn.running_mean, weights, ptr)
                    ptr = self.__load_params(bn.running_var, weights, ptr)
                else:  # If no batch norm then load conv bias
                    ptr = self.__load_params(conv.bias, weights, ptr)
                # Finally, load conv weights
                ptr = self.__load_params(conv.weight, weights, ptr)

    def __load_params(self, param_tensor, param_arr, ptr):
        # Get the number of elements in param_tensor
        num_params = param_tensor.numel()
        # Extract the params from the loaded arr
        params = torch.from_numpy(param_arr[ptr:ptr+num_params])
        params = params.view_as(param_tensor.data)
        # Copy new params to param_tensor
        param_tensor.data.copy_(params)
        # Return increased ptr
        return ptr + num_params
