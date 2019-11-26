import argparse
import time

import imageio
import cv2


from core.models import Darknet
from core.image import draw_predictions

ap = argparse.ArgumentParser()

ap.add_argument("--img", required=True, help="Path to the input image")
ap.add_argument("--cfg", required=True, help="Path to the model config file")
ap.add_argument("-w", "--weights", required=True,
                help="Path to the weights file")
ap.add_argument("--output", help="Path to the output image")
args = ap.parse_args()


# Load input image
img = imageio.imread(args.img)

# Load the model
print("Loading model... ")
net = Darknet(args.cfg)
net.load_weights(args.weights)

# Read COCO label file
with open("data/coco.names") as f:
    content = f.read().strip()
    coco_labels = content.split("\n")

# Inference
start = time.time()
rects, labels, scores = net.detect(img)
print(f"Detection took {time.time() - start:.2f}s")

labels = [coco_labels[i] for i in labels]
res_img = draw_predictions(img, rects, labels, scores)

if args.output:
    print(f"Writing results to {args.output}")
    cv2.imwrite(args.output, res_img)
else:
    cv2.imshow(args.img, res_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
