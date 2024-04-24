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

    intersec[0] = max(j, bbox[0])
    intersec[1] = max(i, bbox[1])
    intersec[2] = min(j + w, bbox[2])
    intersec[3] = min(i + h, bbox[3])
    return intersec


def normalize_intersec(i, j, h, w, intersec):
    '''
    return: normalize into [0, 1]
    '''

    intersec[0] = (intersec[0] - j) / w
    intersec[2] = (intersec[2] - j) / w
    intersec[1] = (intersec[1] - i) / h
    intersec[3] = (intersec[3] - i) / h
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


def draw_bbox(img, gt_box, pre_box, color1=(0, 0, 255), color2=(0, 255, 0)):
    cv2.rectangle(img, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), color1, 2)
    cv2.rectangle(img, (pre_box[0], pre_box[1]), (pre_box[2], pre_box[3]), color2, 2)
    return img


def get_vis(ori_img, img_add, b_gt, b_pr, cam):
    box_img = draw_bbox(img_add.copy(), b_gt, b_pr)
    im_save = np.concatenate([ori_img, img_add, box_img, cam], axis=1)
    return im_save


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
