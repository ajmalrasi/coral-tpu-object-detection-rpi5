import cv2
import base64
import time
import requests

url = 'http://192.168.3.20:8000/predict'

# cap = cv2.VideoCapture("output_video.mp4")
cap = cv2.VideoCapture("videoplayback.mp4")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
reshape = 320
output_video_path = "output_video.mp4"  
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while True:

    start = time.perf_counter()
    ret, frame = cap.read()
    # frame = cv2.imread("/home/affine/Projects/sample_data/2012-12-12_13_55_09.jpg")
    if not ret:
        print("Error reading frame from webcam")
        break
    h, w, c = frame.shape

    # frame_resized = resize_with_padding(frame, desired_size=reshape)
    frame_resized = cv2.resize(frame, (reshape, reshape))

    ret, buffer = cv2.imencode('.jpg', frame_resized)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')

    data = {
        "image": jpg_as_text
    }

    try:
        resp1 = time.perf_counter()
        response = requests.post(url, json=data)
        resp2 = time.perf_counter() - resp1
        # print('Resp time %.2f ms' % (resp2 * 1000))
    except Exception as e:
        print(e.with_traceback())
        raise ValueError("Network Error.")
    
    bbox = list()

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
            ymin = int(ymin * (h / reshape))

            xmax = int(xmax * (w / reshape))
            ymax = int(ymax * (h / reshape))
            bbox.append((xmin, ymin, xmax ,ymax))

            # xmin = int(xmin * (w / reshape))
            # orig = ((reshape / w) * h)
            # rem = (reshape - orig) // 2
            # ymin = int( (ymin - rem) * (h / orig) )
            # xmax = int(xmax * (w / reshape))
            # ymax = int( (ymax  - rem) * (h / orig) )

            text = f"{label}, {score:.2f}"
            cv2.rectangle(frame, (xmin, ymin), ( xmin + len(text) * 8, 
                            ymin - 10) , (255, 255, 255), -1, cv2.LINE_AA)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 255), 1)
            cv2.putText(frame, text, (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX,  0.4, (0, 0, 0), 1, cv2.LINE_AA)
    else:
        print(f"Error sending image: {response.status_code} - {response.text}")

    inference_time = time.perf_counter() - start
    print('Total %.2f ms' % (inference_time * 1000) , 'Resp time %.2f ms' % (resp2 * 1000), end='\r')
    cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
    out.write(frame)
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()