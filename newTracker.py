import math


class Tracker:
    def __init__(self):
        # 存储物体的中心位置
        self.center_points = {}
        # 用于记录目标物体的ID，每次检测到新的物体时，ID会递增
        self.id_count = 0

    def update(self, objects_rect):
        # 存储物体的边界框和ID
        objects_bbs_ids = []

        # 获取新物体的中心点
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # 检查是否已经检测到相同的物体
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 35:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # 如果未检测到相同的物体，为新物体分配ID
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # 清理中心点字典，删除不再使用的ID
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # 更新字典，删除不再使用的ID
        self.center_points = new_center_points.copy()
        return objects_bbs_ids

    # 接收一个带有边界框坐标的numpy数组，
    # 并使用方程xc, yc = (xmin + xmax) // 2, (ymin + ymax) // 2分别计算中心点
    def get_center(self, row):
        center = (int(row[0] + row[2]) // 2, int(row[1] + row[3]) // 2)
        return center
