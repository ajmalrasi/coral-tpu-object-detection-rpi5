import cv2
import requests
import base64
from utils import resize_with_padding

url = 'http://192.168.3.20:8000/predict'


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error reading frame from webcam")
        break
    frame = resize_with_padding(frame, desired_size=300)
    ret, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')

    data = {
        "image": jpg_as_text
    }

    response = requests.post(url, json=data)

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
            text = f"{label} (ID: {id}) - Score: {score:.2f}"

            cv2.putText(frame, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX,  0.7, (0, 255, 0), 2)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    else:
        print(f"Error sending image: {response.status_code} - {response.text}")

    cv2.imshow('Webcam', frame)
    cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()