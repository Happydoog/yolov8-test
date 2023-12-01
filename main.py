import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import *

# 加载 YOLO 模型
model = YOLO('yolov8s.pt')


# 定义鼠标事件回调函数
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        # 打印鼠标指针下的像素值
        colorsBGR = [x, y]
        print(colorsBGR)


# 创建窗口并设置鼠标事件回调函数
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# 打开视频文件
cap = cv2.VideoCapture('veh2.mp4')

# 读取 COCO 类别文件
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

# 初始化计数器
count = 0

# 创建目标追踪器
tracker = Tracker()

# 设置初始参数
cy1 = 322
cy2 = 368
offset = 6

while True:
    # 读取视频帧
    ret, frame = cap.read()
    if not ret:
        break

    # 每隔3帧处理一次
    count += 1
    if count % 3 != 0:
        continue

    # 调整帧大小
    frame = cv2.resize(frame, (1020, 500))

    # 使用 YOLO 模型进行目标检测
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    # 处理检测结果
    list = []
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c:
            list.append([x1, y1, x2, y2])

    # 更新目标追踪器
    bbox_id = tracker.update(list)

    # 在图像上绘制追踪结果
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    # 显示图像
    cv2.imshow("RGB", frame)

    # 检测按键，如果是 'Esc' 键则退出循环
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 释放视频流资源
cap.release()

# 关闭所有窗口
cv2.destroyAllWindows()
