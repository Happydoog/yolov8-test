import sys

sys.path.append('D:\project\python\yolov8counting-trackingvehicles')
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from newTracker import Tracker

import time
from math import dist

model = YOLO(f'D:\project\python\yolov8counting-trackingvehicles\yolov8s.pt')

CONFIDENCE_THRESHOLD = 0.8
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)


cv2.namedWindow('Camera')
cv2.setMouseCallback('Camera', RGB)

cap = cv2.VideoCapture(r'D:\project\python\yolov8counting-trackingvehicles\vedio\A工事存储1_负二层B区-B131_20231111123500_20231111124559.avi')

my_file = open(f"D:\project\python\yolov8counting-trackingvehicles\coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
# print(class_list)

count = 0

tracker = Tracker()

cy1 = 121
cy2 = 340

offset = 6

vh_down = {}
counter = []

vh_up = {}
counter1 = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))
    # 你定义的ROI多边形顶点坐标，这里只是个示例，请替换为实际的坐标值
    roi_vertices = np.array([(72, 326), (710, 66), (840, 70), (739, 435)], np.int32)
    polygon_vertices = roi_vertices.reshape((-1, 1, 2))
    cv2.polylines(frame, [polygon_vertices], isClosed=True, color=(255, 255, 255), thickness=2)

    results = model.predict(frame)
    #   print(results)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    # print(px)
    list = []

    for index, row in px.iterrows():
        xmin = int(row[0])
        ymin = int(row[1])
        xmax = int(row[2])
        ymax = int(row[3])
        xc, yc = tracker.get_center(row)
        # 检查目标的中心点是否在ROI内
        if cv2.pointPolygonTest(np.array(polygon_vertices), (xc, yc), False) >= 0:
            class_id = int(row[5])
            c = class_list[class_id]
            if 'car' in c:
                list.append([xmin, ymin, xmax, ymax])
    bbox_id = tracker.update(list)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx, cy = tracker.get_center(bbox)
        # 绘制车辆框
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 1)
        cv2.putText(frame, str(id), (x3 + 5, y3 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255), 2)
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
