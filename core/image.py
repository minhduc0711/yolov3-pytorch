from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt


def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return np.array(new_image, np.uint8)


def scale_rects(boxes, original_size, input_size):
    oh, ow = original_size
    if isinstance(input_size, tuple):
        input_size = input_size[0]

    # The amount of added gray padding
    pad_w = max(oh - ow, 0) * (input_size / max(original_size))
    pad_h = max(ow - oh, 0) * (input_size / max(original_size))
    # Image dim after padding is removed
    unpad_w = input_size - pad_w
    unpad_h = input_size - pad_h
    # Rescale
    boxes[:, 0] = (boxes[:, 0] - pad_w // 2) / unpad_w * ow
    boxes[:, 1] = (boxes[:, 1] - pad_h // 2) / unpad_h * oh
    boxes[:, 2] = (boxes[:, 2] - pad_w // 2) / unpad_w * ow
    boxes[:, 3] = (boxes[:, 3] - pad_h // 2) / unpad_h * oh
    return boxes.astype(np.int)


def get_colors(n):
    colors = plt.get_cmap("tab20").colors + plt.get_cmap("tab20b").colors + \
        plt.get_cmap("tab20c").colors
    while len(colors) < n:
        colors *= 2
    colors = colors[:n]
    colors = [(int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))
              for c in colors]
    return colors[:n]


def draw_predictions(img, rects, rect_labels, scores,
                     font_scale=0.55, title="predictions"):
    res_img = img.copy()
    colors = get_colors(len(scores))
    for i, (rect, label) in enumerate(zip(rects, rect_labels)):
        # Draw labels
        text = f"{label}: {scores[i]:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_TRIPLEX, font_scale, 2)
        cv2.rectangle(res_img,
                      (rect[0], rect[1] - 10 - text_height),
                      (rect[0] + text_width, rect[1]),
                      colors[i], thickness=cv2.FILLED)
        cv2.putText(res_img, text, (rect[0], rect[1]-10),
                    cv2.FONT_HERSHEY_TRIPLEX, font_scale, (0, 0, 0))
        # Draw bounding box
        cv2.rectangle(res_img, (rect[0], rect[1]),
                      (rect[2], rect[3]), colors[i], 2, cv2.LINE_AA)

    res_img = cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR)
    cv2.imshow(title, res_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
