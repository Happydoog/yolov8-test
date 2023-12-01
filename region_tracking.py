import sys
sys.path.append('D:\project\python\yolov8counting-trackingvehicles')
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import Tracker
import time
from math import dist

model = YOLO(f'D:\project\python\yolov8counting-trackingvehicles\yolov8s.pt')
tracker = Tracker()


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

def get_center(row):
    center = (int(row[0] + row[2]) // 2, int(row[1] + row[3]) // 2)
    return center


cv2.namedWindow('Camera')
cv2.setMouseCallback('Camera', RGB)

cap = cv2.VideoCapture(
    r'D:\project\python\yolov8counting-trackingvehicles\vedio\A工事存储1_负二层B区-B131_20231111123500_20231111124559.avi')

my_file = open(f"D:\project\python\yolov8counting-trackingvehicles\coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
# print(class_list)
count = 0
carCount = 0
cy1 = 326
cy2 = 66
offset = 6
CONFIDENCE_THRESHOLD = 0.8
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
RED = (0, 0, 255)

vh_down = {}
counter = []

vh_up = {}
counter1 = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 6 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))
    # 你定义的ROI多边形顶点坐标，这里只是个示例，请替换为实际的坐标值
    roi_vertices = np.array([(72, 326), (710, 66), (840, 70), (739, 435)], np.int32)
    polygon_vertices = roi_vertices.reshape((-1, 1, 2))
    cv2.polylines(frame, [polygon_vertices], isClosed=True, color=WHITE, thickness=2)

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    list = []

    for index, row in px.iterrows():
        confidence = row[4]
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue
        xmin = int(row[0])
        ymin = int(row[1])
        xmax = int(row[2])
        ymax = int(row[3])
        xc, yc = get_center(row)
        # 检查目标的中心点是否在ROI内
        if cv2.pointPolygonTest(np.array(polygon_vertices), (xc, yc), False) >= 0:
            class_id = int(row[5])
            c = class_list[class_id]
            if 'car' in c:
                list.append([xmin, ymin, xmax, ymax])
                print("检测车辆:", carCount+1, "中心点:", xc, yc, "坐标点:", xmin, ymin, xmax, ymax)
    tracks = tracker.update(list)
    print("车辆数========================", len(tracks))
    for track in tracks:
        # #获取目标id和边界框
        xmin, ymin, xmax, ymax, id = track
    cx, cy = get_center(track)
    print("跟踪车辆:", id, "中心点:", cx, cy, "坐标点:", xmin, ymin, xmax, ymax)
    # 绘制车辆框
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), RED, 1)
    if cy1 < (cy + offset) and cy1 > (cy - offset):
        vh_down[id] = time.time()
    if id in vh_down:
        if cy2 < (cy + offset) and cy2 > (cy - offset):
            if counter.count(id) == 0:
                counter.append(id)
        #####going UP#####
    if cy2 < (cy + offset) and cy2 > (cy - offset):
        vh_up[id] = time.time()
    if id in vh_up:
        if cy1 < (cy + offset) and cy1 > (cy - offset):
            elapsed1_time = time.time() - vh_up[id]
            if counter1.count(id) == 0:
                counter1.append(id)
    cv2.putText(frame, str(id), (xmin + 5, ymin - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2)

    d = (len(counter))
    u = (len(counter1))
    cv2.putText(frame, ('come:-') + str(d), (60, 90), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, ('go:-') + str(u), (60, 130), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
