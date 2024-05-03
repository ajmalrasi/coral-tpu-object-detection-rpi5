import cv2
import requests
import base64

# Server URL (replace with your actual server address and port)
url = 'http://192.168.3.20:8000/predict'

# Initialize video capture from webcam
cap = cv2.VideoCapture(0)  # 0 usually refers to the default webcam

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error reading frame from webcam")
        break
    
    ret, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')

    data = {
        "image": jpg_as_text
    }

    # Send the POST request to the server
    response = requests.post(url, json=data)

    if response.status_code == 200:
        print("Image frame sent successfully")
    else:
        print(f"Error sending image: {response.status_code} - {response.text}")

    # Show the frame (optional, for debugging)
    cv2.imshow('Webcam', frame)
    cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
    # break
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()