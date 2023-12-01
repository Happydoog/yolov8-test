import cv2
from ultralytics import YOLO
import numpy as np
import pandas as pd
from mytrack import track
from poly_area import *
from draw_box_label import box_label
from collections import defaultdict

#                           测试根据左上坐标或坐下坐标判断是否在多边形内
model = YOLO('yolov8l.pt')
my_file = open(f"D:\project\python\yolov8counting-trackingvehicles\coco.txt", "r")
cap = cv2.VideoCapture(
    r'D:\project\python\yolov8counting-trackingvehicles\vedio\A工事存储5_A馆南面机动道路5_20231111123500_20231111124559.avi')
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fNUMS = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videoWriter = cv2.VideoWriter("D:/track/mytrack.mp4", fourcc, fps, size)
data = my_file.read()
class_list = data.split("\n")
# track_history用于保存目标ID，以及它在各帧的目标位置坐标，这些坐标是按先后顺序存储的
track_history = defaultdict(lambda: [])
# 车辆的计数变量
vehicle_in = 0
vehicle_out = 0
CONFIDENCE_THRESHOLD = 0.8
count = 0
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
RED = (0, 0, 255)
CAR_COLOR = (167, 146, 11)
area_poly = [[20, 235], [520, 60], [705, 70], [516, 402]]  # 坐标点
area_poly1 = [[20, 235], [520, 60], [516, 402], [705, 70]]  # 坐标点

bytetracker = track.BYTETracker(fps)


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)


cv2.namedWindow('YOLOv8 Tracking')
cv2.setMouseCallback('YOLOv8 Tracking', RGB)

# 视频帧循环
while cap.isOpened():
    # 读取一帧图像
    success, frame = cap.read()
    if success:
        boxes = []
        confidences = []
        object_classes = []
        frame = cv2.resize(frame, (1020, 500))
        # 定义的ROI多边形顶点坐标，这里只是个示例，请替换为实际的坐标值
        roi_vertices = np.array(area_poly, np.int32)
        polygon_vertices = roi_vertices.reshape((-1, 1, 2))
        cv2.polylines(frame, [polygon_vertices], isClosed=True, color=GREEN, thickness=2)
        # 在帧上运行YOLOv8跟踪，persist为True表示保留跟踪信息，conf为0.3表示只检测置信值大于0.3的目标
        results = model.track(frame, persist=True, conf=0.5)
        outputs = results[0].boxes.data.cpu().numpy()
        for output in outputs:
            x1, y1, x2, y2 = list(map(int, output[:4]))
            if output[6] == 2:
                if is_poi_in_poly([x2, y2], area_poly) or is_poi_in_poly([x1, y1], area_poly):
                    boxes.append([x1, y1, x2, y2])
                    confidences.append(output[5])
                    object_classes.append(output[6])
                    tracks = bytetracker.update(np.array(boxes), np.array(confidences), np.array(object_classes))
                    if len(tracks) > 0:
                        identities = tracks[:, 4]
                        for i, identity in enumerate(identities):
                            box_label(frame, [x1, y1, x2, y2], '#' + str(int(identity)) + ' car', RED)
        # 实时显示进、出车辆的数量
        cv2.putText(frame, 'in: ' + str(vehicle_in), (791, 63),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, 'out: ' + str(vehicle_out), (791, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("YOLOv8 Tracking", frame)  # 显示标记好的当前帧图像
        if cv2.waitKey(1) & 0xFF == 27:  # esc按下时，终止运行
            break
    else:  # 视频播放结束时退出循环
        break

# 释放视频捕捉对象，并关闭显示窗口
cap.release()
cv2.destroyAllWindows()
