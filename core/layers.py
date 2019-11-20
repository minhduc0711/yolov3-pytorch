import numpy as np

import torch
from torch import nn


class YoloLayer(nn.Module):
    """
    Take a prediction tensor of size
        (batch_size, num_anchors * (num_classes + 5), grid_size, grid_size)
    and transform it into a tensor of size
        (batch_size, num_anchors * grid_size * grid_size, num_classes + 5)
    """

    def __init__(self, anchors, num_classes, input_spatial_size):
        super(YoloLayer, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.input_spatial_size = input_spatial_size

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        grid_size = inputs.shape[2]
        strides = self.input_spatial_size // grid_size
        num_anchors = len(self.anchors)
        num_bbox_attrs = self.num_classes + 5

        # Transform the prediction tensor shape
        preds = inputs.view(batch_size, num_anchors,
                            num_bbox_attrs, grid_size ** 2)
        # Becomes (bs, anchors, gs * gs, bbox_attrs)
        preds = preds.transpose(2, 3).contiguous()
        # Becomes (bs, gs * gs, anchors, bbox_attrs)
        preds = preds.transpose(1, 2).contiguous()
        preds = preds.view(batch_size, grid_size *
                           grid_size * num_anchors, num_bbox_attrs)

        # Apply sigmoid to tx, ty, confidence score and class scores
        preds[:, :, 0] = torch.sigmoid(preds[:, :, 0])  # tx
        preds[:, :, 1] = torch.sigmoid(preds[:, :, 1])  # ty
        preds[:, :, 4] = torch.sigmoid(preds[:, :, 4])  # Objectness score
        preds[:, :, 5:] = torch.sigmoid(preds[:, :, 5:])  # Class scores

        # Calculate the absolute x,y of the bounding boxes by
        # offseting top-left coords of corresponding grid
        grid = np.arange(grid_size)
        a, b = np.meshgrid(grid, grid)
        x_offset = torch.FloatTensor(b).repeat_interleave(num_anchors)
        y_offset = torch.FloatTensor(a).repeat_interleave(num_anchors)

        preds[:, :, :2] += torch.stack((x_offset, y_offset), axis=1)
        # Multiply with strides to get correct coords with original image size
        preds[:, :, :2] *= strides

        # Calculate width and height of bounding boxes
        anchors = torch.FloatTensor(self.anchors).repeat((grid_size ** 2, 1))
        preds[:, :, 2:4] = anchors * torch.exp(preds[:, :, 2:4])

        return preds


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


def get_layers(net_params, layer_infos):
    module_list = nn.ModuleList()
    layers_out_channels = []
    input_spatial_size = net_params["width"]

    prev_channels = 3
    for layer_idx, layer_info in enumerate(layer_infos):
        layer_type = layer_info["type"]
        module = nn.Sequential()

        if layer_type == "convolutional":
            out_channels = layer_info["filters"]
            kernel_size = layer_info["size"]
            stride = layer_info["stride"]
            pad = layer_info["pad"]
            activation = layer_info["activation"]

            if "batch_normalize" in layer_info and \
                    layer_info["batch_normalize"] == 1:
                has_bn = True
                bn = nn.BatchNorm2d(out_channels)
                bias = False
            else:
                has_bn = False
                bias = True

            if pad:
                padding = (kernel_size - 1) // 2
            else:
                padding = 0

            conv = nn.Conv2d(in_channels=prev_channels,
                             out_channels=out_channels,
                             kernel_size=kernel_size,
                             stride=stride,
                             padding=padding,
                             bias=bias)

            # Compose 3 layers into a Sequential()
            module.add_module(f"conv_{layer_idx}", conv)
            if activation == "leaky":
                module.add_module(f"leaky_{layer_idx}", nn.LeakyReLU(0.1))
            if has_bn:
                module.add_module(f"batch_norm_{layer_idx}", bn)

        elif layer_type == "maxpool":
            kernel_size = layer_info["size"]
            stride = layer_info["stride"]
            # Strange paddings when stride is 1
            if stride == 1:
                module.add_module(f"padding_{layer_idx}",
                                  nn.ZeroPad2d((0, 1, 0, 1)))
            module.add_module(f"maxpool_{layer_idx}",
                              nn.MaxPool2d(kernel_size=kernel_size,
                                           stride=stride))

        elif layer_type == "upsample":
            factor = layer_info["stride"]
            module.add_module(f"upsample_{layer_idx}",
                              nn.Upsample(scale_factor=factor,
                                          mode='bilinear',
                                          align_corners=False))

        elif layer_type == "route":
            prev_layer_idxs = layer_info["layers"]
            if isinstance(prev_layer_idxs, int):
                prev_layer_idxs = [prev_layer_idxs]
            prev_layer_idxs = [i if i > 0 else layer_idx + i
                               for i in prev_layer_idxs]
            # Update layer indices in the dictionary for convenience
            layer_info["layers"] = prev_layer_idxs

            module.add_module(f"route_{layer_idx}", EmptyLayer())
            out_channels = sum([layers_out_channels[i]
                                for i in prev_layer_idxs])

        elif layer_type == "shortcut":
            # Update layer index
            prev_layer_idx = layer_info["from"]
            prev_layer_idx = prev_layer_idx if prev_layer_idx > 0 \
                else layer_idx + prev_layer_idx
            layer_info["from"] = prev_layer_idx

            module.add_module(f"shortcut_{layer_idx}", EmptyLayer())
            out_channels = layers_out_channels[prev_layer_idx]

        elif layer_type == "yolo":
            # Group anchors values into pairs of (width, height)?? or reverse
            anchors = layer_info["anchors"]
            anchors = [(anchors[i], anchors[i + 1])
                       for i in range(0, len(anchors), 2)]
            # Include only the masked anchors
            anchors = [anchors[i] for i in layer_info["mask"]]
            num_classes = layer_info["classes"]

            module.add_module(f"yolo_{layer_idx}",
                              YoloLayer(anchors=anchors,
                                        num_classes=num_classes,
                                        input_spatial_size=input_spatial_size))

        layers_out_channels.append(out_channels)
        prev_channels = out_channels
        module_list.append(module)

    return module_list
