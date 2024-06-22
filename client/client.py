import cv2
import base64
import time
import numpy as np
import json
import requests
from shapely.geometry import Polygon, box
import random

from sort import *


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        # print(f"Coords: ({x}, {y})",end='\r')
        pass


# url = 'http://192.168.4.100:8000/predict'
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

mot_tracker = Sort() 

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
            # cv2.rectangle(frame, (xmin, ymin), ( xmin + len(text) * 8, 
            #                 ymin - 10) , (255, 255, 255), -1, cv2.LINE_AA)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 255), 1)
            # cv2.putText(frame, text, (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX,  0.4, (0, 0, 0), 1, cv2.LINE_AA)
    else:
        print(f"Error sending image: {response.status_code} - {response.text}")

    overlay = frame.copy()
    # pts = np.array([[[799, 737], [929, 727], [1016, 799], [865, 810]], [[653,746],[786,737],[859,812],[705,818]]], np.int32)
    clrs = (0, 255, 0)


    

# Sample coordinates (adjust as needed)
# polygon_coords = [(0, 0), (5, 0), (5, 4), (2, 5), (0, 2)]
# bounding_box_coords = (1, 1, 4, 3)  # (minx, miny, maxx, maxy)

# polygon = Polygon(polygon_coords)
# bounding_box = box(*bounding_box_coords)

# intersection = polygon.intersection(bounding_box)
# coverage = (intersection.area / polygon.area) * 100

# if coverage >= 80:
#     print("Bounding box covers at least 80% of the polygon.")
# else:
#     print("Bounding box does not cover at least 80% of the polygon.")
    
    track_bbs_ids = mot_tracker.update(np.array(bbox))
    # print(track_bbs_ids[0])

    with open("polygons.json", "r") as file:
        loaded_polygons = json.load(file)
    
    # random.shuffle(loaded_polygons)
        
    num_spots = len(loaded_polygons)

    for pt in loaded_polygons:
        polygon = Polygon(pt)

        cv2.fillPoly(overlay, [np.array(pt)], clrs)


    occupied = 0
    

    for id, bb in zip(track_bbs_ids[:, -1:], track_bbs_ids[:, :4]):

        
        clrs = (0, 255, 0)
        # bounding_box = box(*bb)
        # print(bb)
        minx, miny, maxx, maxy = bb

        text = f"{int(id[0])}, {label}, {score:.2f}"
        cv2.putText(frame, text, (int(minx), int(miny)), cv2.FONT_HERSHEY_COMPLEX,  0.6, (255, 255, 255), 1, cv2.LINE_AA)
        midy =miny + ( ((maxy - miny) // 4) * 2 )
        lower_half_coords = (minx, midy, maxx, maxy) 
        bounding_box = box(*lower_half_coords)

        for pt in loaded_polygons:
            polygon = Polygon(pt)

            intersection = polygon.intersection(bounding_box)
            coverage = (intersection.area / polygon.area) * 100
            if coverage >= 20:
                occupied+=1
                # print("Bounding box covers at least 80% of the polygon.")
                clrs = (0, 0, 255)
                cv2.fillPoly(overlay, [np.array(pt)], clrs)  

                # break
        # break

        
    # for pt, clr in zip(pts, clrs):
    #     print(pt.shape)
    #     cv2.fillPoly(overlay, [pt], clr)  
                

    inference_time = time.perf_counter() - start
    print('Total %.2f ms' % (inference_time * 1000) , 'Resp time %.2f ms' % (resp2 * 1000), end='\r')

    
    cv2.addWeighted(overlay, 0.3, frame, 1 - 0, 0, frame)

    cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)

    cv2.putText(frame, "Occupied {}".format(occupied), (int(100), int(100)), cv2.FONT_HERSHEY_COMPLEX,  0.8, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Free {}".format(num_spots - occupied), (int(100), int(130)), cv2.FONT_HERSHEY_COMPLEX,  0.8, (0, 0, 0), 2, cv2.LINE_AA)

    # cv2.setMouseCallback("Webcam", mouse_callback, param=frame)

    out.write(frame)

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()