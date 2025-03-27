import cv2
import base64
import time
import requests
import os
from datetime import datetime
from utils import resize_with_padding, remap_bbox

# --- Config ---
url = 'http://192.168.3.20:8000/predict'
reshape = 320
use_padding = False  # <-- Set this True to use resize_with_padding

# --- Camera setup ---
cap = cv2.VideoCapture(2)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

os.makedirs("outputs", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
default_path = os.path.join("outputs", f"output_video_{timestamp}.mp4")

# --- Optional video saving (only if --save argument is used) ---
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--save', action='store_true', help="Save output video to disk")
parser.add_argument('--video-path', default=default_path,
                    help="Path to save the output video")

args = parser.parse_args()

out = None
if args.save:
    out = cv2.VideoWriter(args.video_path, fourcc, fps, (width, height))


while True:
    start = time.perf_counter()
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame from webcam")
        break

    h, w, c = frame.shape

    if use_padding:
        frame_resized = resize_with_padding(frame, desired_size=reshape)
    else:
        frame_resized = cv2.resize(frame, (reshape, reshape))

    ret, buffer = cv2.imencode('.jpg', frame_resized)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')

    try:
        resp1 = time.perf_counter()
        response = requests.post(url, json={"image": jpg_as_text})
        resp2 = time.perf_counter() - resp1
    except Exception as e:
        print(e)
        raise ValueError("Network Error.")

    if response.status_code == 200:
        data = response.json()
        for prediction in data["predictions"]:
            bbox = (
                prediction["bbox"]["xmin"],
                prediction["bbox"]["ymin"],
                prediction["bbox"]["xmax"],
                prediction["bbox"]["ymax"],
            )
            label = prediction["label"]
            score = prediction["score"]

            xmin, ymin, xmax, ymax = remap_bbox(bbox, (h, w), reshape, use_padding)

            text = f"{label}, {score:.2f}"
            cv2.rectangle(frame, (xmin, ymin), (xmin + len(text) * 8, ymin - 10), (255, 255, 255), -1, cv2.LINE_AA)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 255), 1)
            cv2.putText(frame, text, (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    else:
        print(f"Error sending image: {response.status_code} - {response.text}")

    inference_time = time.perf_counter() - start
    fps_text = f"FPS: {1.0 / inference_time:.2f}"
    print(f'{fps_text} | Total {inference_time * 1000:.2f} ms | Resp {resp2 * 1000:.2f} ms', end='\r')

    # Show FPS on frame
    cv2.putText(frame, fps_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
    if out:
        out.write(frame)
    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if out:
    out.release()
cv2.destroyAllWindows()
