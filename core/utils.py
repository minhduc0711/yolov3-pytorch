import numpy as np


def cast_to_number(x):
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            return x


def parse_darknet_cfg(fpath):
    with open(fpath, "r") as f:
        content = f.read(-1)
    # Preprocess the lines first
    lines = content.split("\n")
    # Remove comments
    lines = [line for line in lines if not line.startswith("#")]
    lines = [line.strip() for line in lines]  # Strip blank spaces
    lines = [line for line in lines if len(line) > 0]  # Remove empty lines

    # Loop through lines to find blocks
    net_params = None
    layer_infos = []
    i = 0
    while i < len(lines):
        block = {}
        # This first line in the outer loop is always a block type,
        # and we ignore the square brackets
        block["type"] = lines[i][1:-1]

        # Inner loop to extract information of a block
        i += 1
        while i < len(lines):
            line = lines[i]
            if not line.startswith("["):
                key, val = line.split("=")
                vals = val.split(",")
                vals = [v.strip() for v in vals]
                if len(vals) == 1:
                    vals = cast_to_number(vals[0])
                else:
                    try:
                        vals = [cast_to_number(v) for v in vals]
                    except ValueError:
                        pass
                block[key.strip()] = vals
                i += 1
            else:
                break
        if block["type"] == "net":
            net_params = block
        else:
            layer_infos.append(block)
    if net_params is None:
        raise ValueError("[net] block not found")
    return net_params, layer_infos


def get_box_idx(cx, cy, box, grid_size, num_anchors):
    if cx >= grid_size or cy >= grid_size or box >= num_anchors:
        raise ValueError("Values out of bound")
    return cx * grid_size * num_anchors + cy * num_anchors + box


def get_iou(bb1, bb2):
    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0] + 1) * (bb1[3] - bb1[1] + 1)
    bb2_area = (bb2[2] - bb2[0] + 1) * (bb2[3] - bb2[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / \
        float(bb1_area + bb2_area - intersection_area + 1e-16)

    return iou


def xywh2xyxy(rects):
    new_rects = np.zeros_like(rects)

    new_rects[:, 0] = rects[:, 0] - rects[:, 2] / 2
    new_rects[:, 1] = rects[:, 1] - rects[:, 3] / 2
    new_rects[:, 2] = rects[:, 0] + rects[:, 2] / 2
    new_rects[:, 3] = rects[:, 1] + rects[:, 3] / 2

    return new_rects


def non_max_surpression(preds, conf_threshold=0.25, iou_threshold=0.4):
    if not isinstance(preds, np.ndarray):
        preds = np.array(preds)

    # Remove boxes with low confidence score
    idxs = np.where(preds[:, 4] > conf_threshold)[0]
    preds = preds[idxs, :]

    final_rects = []
    final_labels = []
    final_scores = []
    preds[:, :4] = xywh2xyxy(preds[:, :4])
    preds_labels = np.argmax(preds[:, 5:], axis=1)
    # Process boxes of each separate classes to remove redundant ones
    for label in np.unique(preds_labels):
        same_class_rects = preds[preds_labels == label, :4]
        # Calculate P(class) = P(class|object) * P(object)
        same_class_scores = preds[preds_labels == label, 4] * \
            preds[preds_labels == label, 5 + label]
        # Sort the rects by scores descending
        sorted_idxs = np.flip(np.argsort(same_class_scores))
        same_class_rects = same_class_rects[sorted_idxs, :]
        same_class_scores = same_class_scores[sorted_idxs]

        # Surpress bounding boxes that detect the same object,
        # keep only the one with the highest score
        i = 0
        while i < same_class_rects.shape[0]:
            current_rect = same_class_rects[i]
            del_idxs = []  # Keep indices of boxes that will be removed
            # Loop through all other boxes and find big IOU
            for j in range(i + 1, same_class_rects.shape[0]):
                other_rect = same_class_rects[j]
                if get_iou(current_rect, other_rect) > iou_threshold:
                    del_idxs.append(j)
            # Remove the redundant boxes and their corresponding scores
            same_class_rects = np.delete(same_class_rects, del_idxs, axis=0)
            same_class_scores = np.delete(same_class_scores, del_idxs, axis=0)

            i += 1
        # Add the surviving boxes to the final predictions
        final_rects.extend(same_class_rects)
        final_scores.extend(same_class_scores)
        final_labels.extend([label] * same_class_rects.shape[0])

    return np.array(final_rects), final_labels, final_scores
