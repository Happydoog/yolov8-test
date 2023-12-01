from collections import defaultdict

import cv2
from ultralytics import YOLO
import numpy as np
import pandas as pd
from poly_area import in_poly_area_dangerous
from draw_box_label import box_label

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
        frame = cv2.resize(frame, (1020, 500))
        # 定义的ROI多边形顶点坐标，这里只是个示例，请替换为实际的坐标值
        roi_vertices = np.array(area_poly, np.int32)
        polygon_vertices = roi_vertices.reshape((-1, 1, 2))
        cv2.polylines(frame, [polygon_vertices], isClosed=True, color=GREEN, thickness=2)
        # 在帧上运行YOLOv8跟踪，persist为True表示保留跟踪信息，conf为0.3表示只检测置信值大于0.3的目标
        results = model.track(frame, persist=True, conf=0.5)
        # 得到该帧的各个目标的ID
        track_ids = results[0].boxes.id.int().cpu().tolist()
        # 遍历该帧的所有目标
        for track_id, box in zip(track_ids, results[0].boxes.data):
            class_id = int(box[-1])
            c = class_list[class_id]
            if 'car' in c:  # 目标为小汽车
                # 得到该目标矩形框的中心点坐标(x, y)
                x1, y1, x2, y2 = box[:4]
                x = (x1 + x2) / 2
                y = (y1 + y2) / 2
                if in_poly_area_dangerous(box[:4], area_poly):
                    # if cv2.pointPolygonTest(np.array(polygon_vertices), (int(x), int(y)), False) >= 0:
                    # 绘制该目标的矩形框
                    box_label(frame, box, '#' + str(track_id) + ' car', CAR_COLOR)
                    # 提取出该ID的以前所有帧的目标坐标，当该ID是第一次出现时，则创建该ID的字典
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # 追加当前目标ID的坐标
                    # 只有当track中包括两帧以上的情况时，才能够比较前后坐标的先后位置关系
                    if len(track) > 1:
                        _, h = track[-2]  # 提取前一帧的目标纵坐标
                        # 我们设基准线为纵坐标是300的水平线
                        # 前一帧在基准线的上面，当前帧在基准线的下面时，说明该车是从上往下运行
                        if h < 130 <= y:
                            vehicle_in += 1  # out计数加1
                        # 前一帧在基准线的下面，当前帧在基准线的上面时，说明该车是从下往上运行
                        if h > 150 >= y:
                            vehicle_out += 1  # in计数加1
                else:
                    track_history.pop(track_id, [])
        cv2.line(frame, (200, 40), (700, 170), color=(25, 33, 189), thickness=2, lineType=4)
        # 实时显示进、出车辆的数量
        cv2.putText(frame, 'in: ' + str(vehicle_in), (791, 63),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, 'out: ' + str(vehicle_out), (791, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("YOLOv8 Tracking", frame)  # 显示标记好的当前帧图像
        videoWriter.write(frame)
        if cv2.waitKey(1) & 0xFF == 27:  # esc按下时，终止运行
            break
    else:  # 视频播放结束时退出循环
        break

# 释放视频捕捉对象，并关闭显示窗口
cap.release()
videoWriter.release()
cv2.destroyAllWindows()
