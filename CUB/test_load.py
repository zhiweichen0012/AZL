import os
import sys
import json
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.backends import cudnn
import torch.nn as nn
import torchvision
from PIL import Image
import cv2
from utils.func import *
from utils.vis import *
from utils.IoU import *
from utils.augment import *
import argparse
from Model import *
from tqdm import trange
from ptflops import get_model_complexity_info


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Localization evaluation')
        self.parser.add_argument('--input_size', default=256, dest='input_size')
        self.parser.add_argument('--crop_size', default=224, dest='crop_size')
        self.parser.add_argument('--num_classes', default=200)
        self.parser.add_argument('--tencrop', default=True)
        self.parser.add_argument('--phase', type=str, default='test')
        self.parser.add_argument(
            '--gpu', help='which gpu to use', default='0', dest='gpu'
        )
        self.parser.add_argument(
            '--data',
            metavar='DIR',
            default='CUB_DATA/',
            help='path to imagenet dataset',
        )
        self.parser.add_argument(
            '--threshold', type=list, default=[0.24]
        )
        self.parser.add_argument(
            '--top_k',
            type=list,
            default=[130]
        )
        self.parser.add_argument('--arch', type=str, default='vgg')
        self.parser.add_argument('--model_p', type=str, default='./logs')

    def parse(self):
        opt = self.parser.parse_args()
        opt.arch = opt.arch
        return opt


args = opts().parse()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


print("Backbone: ", args.arch)
print("CAM Thresholds: ", args.threshold)


def normalize_map(atten_map, w, h):
    min_val = np.min(atten_map)
    max_val = np.max(atten_map)
    atten_norm = (atten_map - min_val) / (max_val - min_val)
    atten_norm = cv2.resize(atten_norm, dsize=(w, h))
    return atten_norm


def to_data(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data


cudnn.benchmark = True
TEN_CROP = args.tencrop
normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
transform = transforms.Compose(
    [
        transforms.Resize((args.input_size, args.input_size)),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        normalize,
    ]
)
cls_transform = transforms.Compose(
    [
        transforms.Resize((args.input_size, args.input_size)),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        normalize,
    ]
)
ten_crop_aug = transforms.Compose(
    [
        transforms.Resize((args.input_size, args.input_size)),
        transforms.TenCrop(args.crop_size),
        transforms.Lambda(
            lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])
        ),
        transforms.Lambda(
            lambda crops: torch.stack([normalize(crop) for crop in crops])
        ),
    ]
)

model = eval(args.arch).model(args)
model.load_state_dict(torch.load(args.model_p))


print(args)

model = model.to(0)
model.eval()
root = args.data
val_imagedir = os.path.join(root, 'test')

anno_root = os.path.join(root, 'bbox')
val_annodir = os.path.join(root, 'test_gt.txt')
val_list_path = os.path.join(root, 'test_list.txt')

classes = os.listdir(val_imagedir)
classes.sort()

class_to_idx = {classes[i]: i for i in range(len(classes))}

# * Loading gt_box
bbox_f = open(val_annodir, 'r')
bbox_list = []
for line in bbox_f:
    x0, y0, x1, y1, h, w = line.strip("\n").split(' ')
    x0, y0, x1, y1, h, w = (
        float(x0),
        float(y0),
        float(x1),
        float(y1),
        float(h),
        float(w),
    )
    x0, y0, x1, y1 = x0, y0, x1, y1
    bbox_list.append((x0, y0, x1, y1))  ## gt

bbox_f.close()

files = [[] for i in range(200)]

with open(val_list_path, 'r') as f:
    for line in f:
        test_img_path, img_class = line.strip("\n").split(';')
        files[int(img_class)].append(test_img_path)

for top_k in args.top_k:
    # * Init
    cur_num = 0
    accs = {}
    accs_top5 = {}
    loc_accs = {}
    AccTop1 = []
    AccTop5 = []

    IoUSet = {}
    IoUSetTop5 = {}
    LocSet = {}
    BoxesPred = {}

    # * range args.threshold
    for _th in args.threshold:
        accs[str(_th)] = []
        accs_top5[str(_th)] = []
        loc_accs[str(_th)] = []

    for k in trange(200):
        cls = classes[k]

        for _th in args.threshold:
            IoUSet[str(_th)] = []
            IoUSetTop5[str(_th)] = []
            LocSet[str(_th)] = []
            BoxesPred[str(_th)] = []

        for (i, name) in enumerate(files[k]):

            gt_boxes = bbox_list[cur_num]
            cur_num += 1
            if len(gt_boxes) == 0:
                continue

            raw_img = Image.open(os.path.join(val_imagedir, name)).convert('RGB')
            w, h = args.crop_size, args.crop_size

            with torch.no_grad():
                img = transform(raw_img)
                img = torch.unsqueeze(img, 0)
                img = img.to(0)
                output_dict = model(img, torch.tensor([class_to_idx[cls]]), top_k)

                cam = model.x_saliency[0][0].data.cpu()
                cam = normalize_map(np.array(cam), w, h)

                # * Generate box
                for _th in args.threshold:
                    # BoxesPred[str(_th)] = get_PBox(cam=cam, w=w, h=h, _th=_th)
                    BoxesPred[str(_th)] = get_CBox(cam=cam, w=w, h=h, _th=_th)

                if TEN_CROP:
                    img = ten_crop_aug(raw_img)
                    img = img.to(0)
                    output_dict = model(img, torch.tensor(class_to_idx[cls]).expand(10))
                    vgg16_out = output_dict['score_1']
                    vgg16_out = nn.Softmax(dim=1)(vgg16_out)
                    vgg16_out = torch.mean(vgg16_out, dim=0, keepdim=True)
                    vgg16_out = torch.topk(vgg16_out, 5, 1)[1]
                else:
                    img = cls_transform(raw_img)
                    img = torch.unsqueeze(img, 0)
                    img = img.to(0)
                    output_dict = model(img, [class_to_idx[cls]])
                    vgg16_out = output_dict['score_1']
                    vgg16_out = torch.topk(vgg16_out, 5, 1)[1]
                vgg16_out = to_data(vgg16_out)
                vgg16_out = torch.squeeze(vgg16_out)
                vgg16_out = vgg16_out.numpy()
                out = vgg16_out

            # handle resize and centercrop for gt_boxes

            gt_bbox_i = list(gt_boxes)
            raw_img_i = raw_img
            raw_img_i, gt_bbox_i = ResizedBBoxCrop((256, 256))(raw_img, gt_bbox_i)
            raw_img_i, gt_bbox_i = CenterBBoxCrop((224))(raw_img_i, gt_bbox_i)
            # w, h = raw_img_i.size
            gt_bbox_i[0] = gt_bbox_i[0] * w
            gt_bbox_i[2] = gt_bbox_i[2] * w
            gt_bbox_i[1] = gt_bbox_i[1] * h
            gt_bbox_i[3] = gt_bbox_i[3] * h
            gt_boxes = gt_bbox_i

            # * LOC
            for _th in args.threshold:
                max_iou = -1
                iou = IoU(BoxesPred[str(_th)], gt_boxes)
                if iou > max_iou:
                    max_iou = iou

                LocSet[str(_th)].append(max_iou)
                temp_loc_iou = max_iou
                if out[0] != class_to_idx[cls]:
                    max_iou = 0

                IoUSet[str(_th)].append(max_iou)
                # cal top5 IoU
                max_iou = 0
                for i in range(5):
                    if out[i] == class_to_idx[cls]:
                        max_iou = temp_loc_iou
                IoUSetTop5[str(_th)].append(max_iou)
            # * CLS
            if out[0] == class_to_idx[cls]:
                AccTop1.append(1.0)
            else:
                AccTop1.append(0.0)
            for i in range(5):
                if out[i] == class_to_idx[cls]:
                    AccTop5.append(1.0)
                    break

        for _th in args.threshold:
            cls_loc_acc = np.sum(np.array(IoUSet[str(_th)]) > 0.5) / len(
                IoUSet[str(_th)]
            )
            cls_loc_acc_top5 = np.sum(np.array(IoUSetTop5[str(_th)]) > 0.5) / len(
                IoUSetTop5[str(_th)]
            )
            loc_acc = np.sum(np.array(LocSet[str(_th)]) > 0.5) / len(LocSet[str(_th)])
            # TODO cal ALL
            accs[str(_th)].append(cls_loc_acc)
            accs_top5[str(_th)].append(cls_loc_acc_top5)
            loc_accs[str(_th)].append(loc_acc)

    best_top1Loc = 0.0
    best_top5Loc = 0.0
    best_gt = 0.0
    best_th = 0.0

    for _th in args.threshold:
        if np.mean(accs[str(_th)]) * 100 > best_top1Loc:
            best_top1Loc = np.mean(accs[str(_th)]) * 100
            best_top5Loc = np.mean(accs_top5[str(_th)]) * 100
            best_gt = np.mean(loc_accs[str(_th)]) * 100
            best_th = _th

    print()
    print("-" * 30)
    print("------ Best Performance ------->TOP-{}".format(top_k))
    print("-" * 30)
    print(
        '* Loc Acc@1 \033[32m{top1:.3f}\033[0m Acc@5 {top5:.3f} GT \033[32m{gt:.3f}\033[0m TH \033[4m{th:.3f}\033[0m TestNum {tn:d}'.format(
            top1=best_top1Loc,
            top5=best_top5Loc,
            gt=best_gt,
            th=best_th,
            tn=len(AccTop1),
        )
    )
    print(
        '* Acc@1 \033[32m{top1:.3f} \033[0m Acc@5 {top5:.3f}'.format(
            top1=np.sum(np.array(AccTop1)) / len(AccTop1) * 100,
            top5=np.sum(np.array(AccTop5)) / len(AccTop1) * 100,
        )
    )
    print(
        "{top1_loc:.3f} {top5_loc:.3f} {gt:.3f} {th:.3f} {top1_cls:.3f} {top5_cls:.3f}".format(
            top1_loc=best_top1Loc,
            top5_loc=best_top5Loc,
            gt=best_gt,
            th=best_th,
            top1_cls=np.sum(np.array(AccTop1)) / len(AccTop1) * 100,
            top5_cls=np.sum(np.array(AccTop5)) / len(AccTop1) * 100,
        )
    )
    
