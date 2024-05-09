import cv2
import requests
import base64
from utils import resize_with_padding
import time


url = 'http://192.168.3.20:8000/predict'

cap = cv2.VideoCapture("cam1.mkv")

reshape = 300

while True:
    ret, frame = cap.read()

    # frame = cv2.imread("/home/affine/Projects/sample_data/2012-12-12_13_55_09.jpg")

    if not ret:
        print("Error reading frame from webcam")
        break
    start = time.perf_counter()

    h, w, c = frame.shape

    frame_resized = resize_with_padding(frame, desired_size=reshape)

    ret, buffer = cv2.imencode('.jpg', frame_resized)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')

    data = {
        "image": jpg_as_text
    }

    try:
        response = requests.post(url, json=data)
    except Exception as e:
        print(e.with_traceback())
        raise ValueError("Network Error.")

    if response.status_code == 200:
        data = response.json()
        for prediction in data["predictions"]:
            xmin = prediction["bbox"]["xmin"]
            ymin = prediction["bbox"]["ymin"] 
            xmax = prediction["bbox"]["xmax"]
            ymax = prediction["bbox"]["ymax"]
            label = prediction["label"] 
            score = prediction["score"]
            id = prediction["id"]

            xmin = int(xmin * (w / reshape))
            orig = ((reshape / w) * h)
            rem = (reshape - orig) // 2
            ymin = int( (ymin - rem) * (h / orig) )
            xmax = int(xmax * (w / reshape))
            ymax = int( (ymax  - rem) * (h / orig) )

            text = f"{label}, {score:.2f}"
            cv2.rectangle(frame, (xmin, ymin), ( xmin + len(text) * 8, 
                            ymin - 10) , (0, 255, 0), -1, cv2.LINE_AA)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, text, (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX,  0.4, (0, 0, 0), 1, cv2.LINE_AA)
    else:
        print(f"Error sending image: {response.status_code} - {response.text}")
    inference_time = time.perf_counter() - start
    print('%.2f ms' % (inference_time * 1000))

    cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()