import cv2
import base64
import time
import numpy as np


lliisstt = []


def get_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        coordinates = [x, y]
        lliisstt.append(coordinates)
        print(f"Clicked coordinates: {coordinates}")


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Coords: ({x}, {y})", end='\r')


cap = cv2.VideoCapture("/media/ajmalrasi/dev/projects/rpi/coral-tpu-object-detection-rpi5/client/videoplayback.mp4")
reshape = 320

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame from video")
        break
    
    start = time.perf_counter()
    h, w, c = frame.shape
    frame_resized = cv2.resize(frame, (reshape, reshape))
    ret, buffer = cv2.imencode('.jpg', frame_resized)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    data = {"image": jpg_as_text}

    
    sliced_list = [lliisstt[i:i + 4] for i in range(0, len(lliisstt), 4)]

    overlay = frame.copy()
    pts = [np.array(pt, np.int32) for pt in sliced_list if len(pt) == 4]
    clrs = [(0, 0, 255), (0, 255, 0), (0, 255, 0), (0, 255, 0)]
    
    for pt, clr in zip(pts, clrs):
        cv2.fillPoly(overlay, [pt], clr)
    cv2.addWeighted(overlay, 0.3, frame, 1 - 0.3, 0, frame)
    
    cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Webcam', mouse_callback, param=frame)
    cv2.setMouseCallback('Webcam', get_coordinates, param=frame)
    
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(sliced_list)