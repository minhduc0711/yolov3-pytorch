import argparse
import time

from PIL import Image
import numpy as np
import torch


from core.models import Darknet
from core.utils import parse_darknet_cfg, non_max_surpression
from core.image import (
    draw_predictions, letterbox_image, scale_rects, get_colors
)

ap = argparse.ArgumentParser()

ap.add_argument("img", help="Path to the input image")
ap.add_argument("cfg", help="Path to the model config file")
ap.add_argument("weights", help="Path to the weights file")

args = ap.parse_args()

# Load the model
print("Loading model... ")
net_params, layer_infos = parse_darknet_cfg(args.cfg)
net = Darknet(net_params, layer_infos)
input_dim = net_params["height"], net_params["width"]
net.load_weights(args.weights)

# Load and prepare the input image
img = Image.open(args.img)
# Pad & resize
resized_img = letterbox_image(img, input_dim)
img = np.array(img)
net_input = torch.from_numpy(np.transpose(
    resized_img, [2, 0, 1])).type(torch.float32)
net_input /= 255.
net_input = net_input.unsqueeze(0)

# Read COCO label file
with open("data/coco.names") as f:
    content = f.read().strip()
    coco_labels = content.split("\n")
    colors = get_colors(len(coco_labels))

# Inference
start = time.time()
with torch.no_grad():
    net.eval()
    net_output = net(net_input)

rects, labels, scores = non_max_surpression(
    net_output[0], conf_threshold=0.25, iou_threshold=0.4)
print(f"Detection took {time.time() - start:.2f}s")

rects = scale_rects(rects, img.shape[:2], input_dim)
labels = [coco_labels[i] for i in labels]
draw_predictions(img, rects, labels, scores, title=args.img)
