import argparse
from tqdm import tqdm

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
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=num_frames)
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
while True:
    ret, frame = cap.read()
    if not ret:
        break
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rects, labels, scores = net.detect(img)
    if len(rects) > 0:
        res_img = draw_predictions(img, rects, labels, scores, coco_labels)
    else:
        res_img = img
    res_img = cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR)

    if is_webcam:
        cv2.imshow("real time YOLOv3", res_img)
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        if writer is None:
            writer = cv2.VideoWriter(
                output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
        writer.write(res_img)
        pbar.update(1)

pbar.close()
if writer is not None:
    writer.release()
    print(f"Output video saved to {output_path}")
cap.release()
cv2.destroyAllWindows()
