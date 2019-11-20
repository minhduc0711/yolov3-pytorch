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
