import argparse

import cv2

from core.models import Darknet
from core.image import draw_predictions

ap = argparse.ArgumentParser()

ap.add_argument("--vid",
                help="Path to the input video, or set to 0 for webcam")
ap.add_argument("--cfg", required=True, help="Path to the model config file")
ap.add_argument("-w", "--weights", required=True,
                help="Path to the weights file")
args = ap.parse_args()

if args.vid == "0":
    is_webcam = True
    cap = cv2.VideoCapture(0)
else:
    is_webcam = False
    cap = cv2.VideoCapture(args.vid)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = "output/detections.mp4"
writer = None

# Load the model
print("Loading model... ")
net = Darknet(args.cfg)
net.load_weights(args.weights)

# Read COCO label file
with open("data/coco.names") as f:
    content = f.read().strip()
    coco_labels = content.split("\n")

print("Detecting...")
frame_cnt = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_cnt += 1
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    rects, labels, scores = net.detect(img)
    if len(rects) > 0:
        labels = [coco_labels[i] for i in labels]
        res_img = draw_predictions(img, rects, labels, scores)
    else:
        res_img = img
    res_img = cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR)

    if is_webcam:
        cv2.imshow(args.vid, res_img)
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        if writer is None:
            writer = cv2.VideoWriter(
                output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
        if frame_cnt % 100 == 0:
            print(f"Processing frame {frame_cnt}")
        writer.write(res_img)

if writer is not None:
    writer.release()
cap.release()
cv2.destroyAllWindows()
