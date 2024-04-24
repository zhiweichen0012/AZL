import numpy as np
import matplotlib.pyplot as plt
import torch
import copy
from skimage import measure
import cv2
import os


def count_max(x):
    count_dict = {}
    for xlist in x:
        for item in xlist:
            if item == 0:
                continue
            if item not in count_dict.keys():
                count_dict[item] = 0
            count_dict[item] += 1
    if count_dict == {}:
        return -1
    count_dict = sorted(count_dict.items(), key=lambda d: d[1], reverse=True)
    return count_dict[0][0]


def compute_intersec(i, j, h, w, bbox):
    '''
    intersection box between croped box and GT BBox
    '''
    intersec = copy.deepcopy(bbox)
    box_num = len(bbox) // 4
    for x in range(box_num):
        intersec[0 + 4 * x] = max(j, bbox[0 + 4 * x])
        intersec[1 + 4 * x] = max(i, bbox[1 + 4 * x])
        intersec[2 + 4 * x] = min(j + w, bbox[2 + 4 * x])
        intersec[3 + 4 * x] = min(i + h, bbox[3 + 4 * x])
    return intersec


def normalize_intersec(i, j, h, w, intersec):
    '''
    return: normalize into [0, 1]
    '''
    box_num = len(intersec) // 4
    for x in range(box_num):
        intersec[0 + 4 * x] = (intersec[0 + 4 * x] - j) / w
        intersec[2 + 4 * x] = (intersec[2 + 4 * x] - j) / w
        intersec[1 + 4 * x] = (intersec[1 + 4 * x] - i) / h
        intersec[3 + 4 * x] = (intersec[3 + 4 * x] - i) / h
    return intersec


def get_PBox(cam, w, h, _th):
    highlight = np.zeros(cam.shape)
    highlight[cam > _th] = 1
    # max component
    all_labels = measure.label(highlight)
    highlight = np.zeros(highlight.shape)
    highlight[all_labels == count_max(all_labels.tolist())] = 1
    highlight = np.round(highlight * 255)
    highlight_big = cv2.resize(highlight, (w, h), interpolation=cv2.INTER_NEAREST)
    CAMs = copy.deepcopy(highlight_big)
    props = measure.regionprops(highlight_big.astype(int))

    if len(props) == 0:
        bbox = [0, 0, w, h]
    else:
        temp = props[0]['bbox']
        bbox = [temp[1], temp[0], temp[3], temp[2]]
    return bbox


def get_CBox(cam, _th, w=None, h=None):
    """
    cam: single image with shape (h, w, 1)
    thr_val: float value (0~1)
    return estimated bounding box
    """
    cam = (cam * 255.0).astype(np.uint8)
    map_thr = _th * np.max(cam)

    _, thr_gray_heatmap = cv2.threshold(cam, int(map_thr), 255, cv2.THRESH_TOZERO)

    contours, _ = cv2.findContours(
        thr_gray_heatmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        estimated_bbox = [x, y, x + w, y + h]
    else:
        estimated_bbox = [0, 0, 1, 1]

    return estimated_bbox  # , thr_gray_heatmap, len(contours)


def getFileList(dir, Filelist, ext=None):
    newDir = dir
    if os.path.isfile(dir):
        if ext is None:
            Filelist.append(dir)
        else:
            if ext in dir[-10:]:
                Filelist.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            getFileList(newDir, Filelist, ext)
    return Filelist
