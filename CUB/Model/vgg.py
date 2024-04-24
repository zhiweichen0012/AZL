import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import cv2
from skimage import measure
from utils.func import *


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.num_classes = args.num_classes
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  ## -> 64x224x224
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  ## -> 64x224x224
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2)  ## -> 64x112x112

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  ## -> 128x112x112
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)  ## -> 128x112x112
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2)  ## -> 128x56x56

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  ## -> 256x56x56
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)  ## -> 256x56x56
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)  ## -> 256x56x56
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2)  ## -> 256x28x28

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)  ## -> 512x28x28
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)  ## -> 512x28x28
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)  ## -> 512x28x28
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2)  ## -> 512x14x14

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)  ## -> 512x14x14
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)  ## -> 512x14x14
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)  ## -> 512x14x14
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2)

        self.avg_pool = nn.AvgPool2d(14)  ## ->(512,1,1)

        self.classifier_cls = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 200, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.classifier_loc = nn.Sequential(
            nn.Conv2d(512, 200, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x, label=None, N=1):
        conv_copy_5_1 = copy.deepcopy(self.conv5_1)
        relu_copy_5_1 = copy.deepcopy(self.relu5_1)
        conv_copy_5_2 = copy.deepcopy(self.conv5_2)
        relu_copy_5_2 = copy.deepcopy(self.relu5_2)
        conv_copy_5_3 = copy.deepcopy(self.conv5_3)
        relu_copy_5_3 = copy.deepcopy(self.relu5_3)
        classifier_cls_copy = copy.deepcopy(self.classifier_cls)

        batch = x.size(0)
        x = self.conv1_1(x)
        x = self.relu1_1(x)
        x = self.conv1_2(x)
        x = self.relu1_2(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.relu2_1(x)
        x = self.conv2_2(x)
        x = self.relu2_2(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.relu3_1(x)
        x = self.conv3_2(x)
        x = self.relu3_2(x)
        x = self.conv3_3(x)
        x = self.relu3_3(x)

        x = self.pool3(x)
        x = self.conv4_1(x)
        x = self.relu4_1(x)
        x = self.conv4_2(x)
        x = self.relu4_2(x)
        x = self.conv4_3(x)
        x = self.relu4_3(x)
        x_4 = x.clone()

        x = self.pool4(x)

        x = self.conv5_1(x)
        x = self.relu5_1(x)
        x = self.conv5_2(x)
        x = self.relu5_2(x)
        x = self.conv5_3(x)
        x = self.relu5_3(x)

        x = self.classifier_cls(x)
        self.feature_map = x

        # * CLS1
        self.score_1 = self.avg_pool(x).view(x.size(0), -1)

        # * label
        if N == 1:
            p_label = label.unsqueeze(-1)
        else:
            _, p_label = self.score_1.topk(N, 1, True, True)

        # * LOC
        x_saliency_all = self.classifier_loc(x_4)
        x_saliency = torch.zeros(batch, 1, 28, 28).cuda()
        for i in range(batch):
            x_saliency[i][0] = x_saliency_all[i][p_label[i]].mean(0)
        self.x_saliency = x_saliency

        # * CLS2
        x_fg = self.feature_map * nn.AvgPool2d(2)(self.x_saliency)
        self.score_2 = self.avg_pool(x_fg).view(x_fg.size(0), -1)

        # TODO active activation
        x_erase = x_4.detach() * (1.0 - self.x_saliency)

        x_erase = self.pool4(x_erase)
        x_erase = conv_copy_5_1(x_erase)
        x_erase = relu_copy_5_1(x_erase)
        x_erase = conv_copy_5_2(x_erase)
        x_erase = relu_copy_5_2(x_erase)
        x_erase = conv_copy_5_3(x_erase)
        x_erase = relu_copy_5_3(x_erase)
        x_erase = classifier_cls_copy(x_erase)

        x_enhance = self.feature_map - x_erase

        self.score_bg = self.avg_pool(x_erase).view(x_erase.size(0), -1)
        self.score_fg = self.avg_pool(x_enhance).view(x.size(0), -1)

        self.x_erase_sum = torch.zeros(batch).cuda()
        for i in range(batch):
            self.x_erase_sum[i] = self.score_bg[i][label[i]]

        self.enhance_sum = torch.zeros(batch).cuda()
        for i in range(batch):
            self.enhance_sum[i] = self.score_fg[i][label[i]]

        return {
            'score_1': self.score_1,
            'score_2': self.score_2,
        }

    def bf_loss(self):
        enhance_sum = self.enhance_sum.clone().detach()
        x_res = self.x_erase_sum
        res = x_res / (enhance_sum + 1e-8)
        res[x_res >= enhance_sum] = 0
        return res.mean(0)

    def ac_loss(self):
        batch = self.x_saliency.size(0)
        x_saliency = self.x_saliency
        x_saliency = x_saliency.clone().view(batch, -1)
        return x_saliency.mean()

    def active_loss(self, label=None, min_area=0, max_area=1.0):
        batch_size = self.x_saliency.shape[0]
        min_prob = 1e-16
        losses = torch.zeros(batch_size).cuda()
        fg_probs = F.softmax(self.score_fg, dim=1).clamp(min=min_prob)
        bg_probs = F.softmax(self.score_bg, dim=1).clamp(min=min_prob)
        for i in range(batch_size):
            mask = self.x_saliency[i].flatten()
            sorted_mask, indices = mask.sort(descending=True)
            losses[i] = (
                self._min_mask_area_loss(sorted_mask=sorted_mask, min_area=min_area)
                + self._max_mask_area_loss(sorted_mask=sorted_mask, max_area=max_area)
            ).mean()
            if bg_probs[i][label[i]] > fg_probs[i][label[i]]:
                losses[i] = losses[i] + self.args.lambda_e * self._max_min_active_loss(
                    sorted_mask=sorted_mask, min_area=min_area, max_area=max_area
                )
        return torch.mean(losses)

    def _min_mask_area_loss(self, sorted_mask, min_area):
        ones_length = (int)(28 * 28 * min_area)
        ones = torch.ones(ones_length).cuda()
        zeros = torch.zeros((28 * 28) - ones_length).cuda()
        ones_and_zeros = torch.cat((ones, zeros), dim=0)
        # [1, 1, 0, 0, 0] - [0.9, 0.9, 0.9, 0.5, 0.1] = [0.1, 0.1, -0.9, -0.5, -0.1] -> [0.1, 0.1, 0, 0, 0]
        loss = F.relu(ones_and_zeros - sorted_mask, inplace=True)
        return loss

    def _max_mask_area_loss(self, sorted_mask, max_area):
        ones_length = (int)(28 * 28 * max_area)
        ones = torch.ones(ones_length).cuda()
        zeros = torch.zeros((28 * 28) - ones_length).cuda()
        ones_and_zeros = torch.cat((ones, zeros), dim=0)
        # [0.9, 0.9, 0.9, 0.5, 0.1] - [1, 1, 1, 1, 0] = [-0.1, -0.1, -0.1, -0.5, 0.1] -> [0, 0, 0, 0, 0.1]
        loss = F.relu(sorted_mask - ones_and_zeros, inplace=True)
        return loss

    def _max_min_active_loss(self, sorted_mask, min_area, max_area):
        zeros_length_max = (int)(28 * 28 * max_area)
        zeros_length_min = (int)(28 * 28 * min_area)
        zeros_max = torch.zeros(zeros_length_max).cuda()
        zeros_min = torch.zeros(zeros_length_min).cuda()
        ones = torch.ones((28 * 28) - zeros_length_max - zeros_length_min).cuda()
        ones_and_zeros = torch.cat((zeros_max, ones, zeros_min), dim=0)
        res = torch.mean(sorted_mask * ones_and_zeros).clamp(max=1.0)
        return -torch.log(res)


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0, 0.01)
        m.weight.data.fill_(0)


def model(args, pretrained=True):

    model = Model(args)
    if pretrained:
        model.apply(weight_init)
        pretrained_dict = torch.load('../weights/vgg16-397923af.pth')
        model_dict = model.state_dict()
        model_conv_name = []

        for i, (k, v) in enumerate(model_dict.items()):
            model_conv_name.append(k)
        for i, (k, v) in enumerate(pretrained_dict.items()):
            if k.split('.')[0] != 'features':
                break
            if np.shape(model_dict[model_conv_name[i]]) == np.shape(v):
                model_dict[model_conv_name[i]] = v
        model.load_state_dict(model_dict)
        print("Loading pretrained weight Complete...")
    return model
