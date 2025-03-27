import cv2


def resize_with_padding(img, desired_size=300):
    h, w = img.shape[:2]
    aspect_ratio = w / h
    if h >= w:
        new_h = desired_size
        new_w = int(new_h * aspect_ratio)
    else:
        new_w = desired_size
        new_h = int(new_w / aspect_ratio)

    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_w = (desired_size - new_w) // 2
    pad_h = (desired_size - new_h) // 2

    return cv2.copyMakeBorder(resized_img, top=pad_h, bottom=pad_h, 
                              left=pad_w, right=pad_w, 
                              borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])



def remap_bbox(bbox, original_shape, reshape, use_padding):
    h, w = original_shape
    xmin, ymin, xmax, ymax = bbox

    if use_padding:
        orig_h = int((reshape / w) * h)
        rem = (reshape - orig_h) // 2
        xmin = int(xmin * (w / reshape))
        ymin = int((ymin - rem) * (h / orig_h))
        xmax = int(xmax * (w / reshape))
        ymax = int((ymax - rem) * (h / orig_h))
    else:
        xmin = int(xmin * (w / reshape))
        ymin = int(ymin * (h / reshape))
        xmax = int(xmax * (w / reshape))
        ymax = int(ymax * (h / reshape))

    return xmin, ymin, xmax, ymax