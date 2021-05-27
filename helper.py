# coding=utf-8
import os

import cv2
import geometry_msgs.msg
from matplotlib import pyplot as plt
import numpy as np

path_root = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), './'))

def plt_show(img):
    if len(img.shape) == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img
    plt.imshow(img_rgb), plt.colorbar(), plt.show()


def py_cpu_nms(bboxes, scores, thresh=0.5):
    """Pure Python NMS baseline."""
    # x1、y1、x2、y2、以及score赋值
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    # 每一个检测框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # remove small area
    large_areas = np.where(areas > 1000)[0]

    # 按照score置信度降序排序
    order = scores.argsort()[::-1]
    order = np.intersect1d(large_areas, order)
    keep = []
    # 保留的结果框集合
    while order.size > 0:
        i = order[0]
        keep.append(i)  # 保留该类剩余box中得分最高的一个
        # 得到相交区域,左上及右下
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 计算相交的面积,不重叠时面积为0
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 计算IoU：重叠面积 /（面积1+面积2-重叠面积）
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # 保留IoU小于阈值的box
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
        # 因为ovr数组的长度比order数组少一个,所以这里要将所有下标后移一位
    return keep


def bboxes_extend(bboxes, ratio=0.05):
    bboxex_ext=bboxes.copy()
    width = bboxes[:, 2] - bboxes[:, 0]
    height = bboxes[:, 3] - bboxes[:, 1]
    width_ext = (width * ratio / 2).astype(int)
    height_ext = (height * ratio / 2).astype(int)
    bboxex_ext[:, 0] = np.maximum(0, bboxes[:, 0] - width_ext)
    bboxex_ext[:, 2] = np.minimum(640 - 1, bboxes[:, 2] + width_ext)
    bboxex_ext[:, 1] = np.maximum(0, bboxes[:, 1] - height_ext)
    bboxex_ext[:, 3] = np.minimum(480 - 1, bboxes[:, 3] + height_ext)
    return bboxex_ext

def gen_pose(x,y,z,rx,ry,rz,rw):
    pose=geometry_msgs.msg.Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    pose.orientation.x = rx
    pose.orientation.y = ry
    pose.orientation.z = rz
    pose.orientation.w = rw
    return pose

def save_file(str, filepath):
    with  open(filepath, "w") as file:
        file.write(str)