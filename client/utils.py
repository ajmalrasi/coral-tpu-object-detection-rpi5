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
