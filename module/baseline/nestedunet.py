import numpy as np

from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision
from simplecv.interface import CVModule
from simplecv import registry
from torch.utils.checkpoint import checkpoint


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, act_func=nn.ReLU(inplace=True)):
        super(VGGBlock, self).__init__()
        self.act_func = act_func
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_func(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act_func(out)

        return out


class UNet(CVModule):
    def __init__(self, config):
        super(UNet, self).__init__(config)

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(3, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], self.config.num_classes, kernel_size=1)


    def forward(self, x, y=None):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        cls_pred = self.final(x0_4)

        if self.training:
            cls_true = y['cls']
            loss_dict = {
                'cls_loss': self.cls_loss(cls_pred, cls_true)
            }
            return loss_dict

        cls_prob = torch.softmax(cls_pred, dim=1)
        return cls_prob

    def cls_loss(self, y_pred, y_true):
        loss = F.cross_entropy(y_pred, y_true.long(), ignore_index=self.config.loss.ignore_index)

        return loss

    def set_defalut_config(self):
        self.config.update(dict(
            num_classes=16,
            loss=dict(
                ignore_index=255
            )
        ))

@registry.MODEL.register('NestedUNet')
class NestedUNet(CVModule):
    def __init__(self, config):
        super(NestedUNet, self).__init__(config)


        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(3, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.config.deepsupervision:
            self.final1 = nn.Conv2d(nb_filter[0], self.config.num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], self.config.num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], self.config.num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], self.config.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], self.config.num_classes, kernel_size=1)

    def nest1(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        return x0_0, x1_0, x0_1

    def nest2(self, x0_0, x1_0, x0_1):
        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        return x2_0, x1_1, x0_2

    def nest3(self, x0_0, x1_0, x0_1, x2_0, x1_1, x0_2):
        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        return x3_0, x2_1, x1_2, x0_3

    def nest4(self, x0_0, x1_0, x0_1, x2_0, x1_1, x0_2, x3_0, x2_1, x1_2, x0_3):
        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        return x0_4

    def forward(self, x, y=None):
        if self.config.cp[0] and x.requires_grad:
            x0_0, x1_0, x0_1 = checkpoint(self.nest1, x)
        else:
            x0_0, x1_0, x0_1 = self.nest1(x)
        # x0_0 = self.conv0_0(x)
        # x1_0 = self.conv1_0(self.pool(x0_0))
        # x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        # x2_0 = self.conv2_0(self.pool(x1_0))
        # x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        # x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        if self.config.cp[1]:
            x2_0, x1_1, x0_2 = checkpoint(self.nest2, x0_0, x1_0, x0_1)
        else:
            x2_0, x1_1, x0_2 = self.nest2(x0_0, x1_0, x0_1)


        # x3_0 = self.conv3_0(self.pool(x2_0))
        # x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        # x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        # x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        if self.config.cp[2]:
            x3_0, x2_1, x1_2, x0_3 = checkpoint(self.nest3, x0_0, x1_0, x0_1, x2_0, x1_1, x0_2)
        else:
            x3_0, x2_1, x1_2, x0_3 = self.nest3(x0_0, x1_0, x0_1, x2_0, x1_1, x0_2)
        # x4_0 = self.conv4_0(self.pool(x3_0))
        # x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        # x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        # x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        # x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        if self.config.cp[3]:
            x0_4 = checkpoint(self.nest4, x0_0, x1_0, x0_1, x2_0, x1_1, x0_2, x3_0, x2_1, x1_2, x0_3)
        else:
            x0_4 = self.nest4(x0_0, x1_0, x0_1, x2_0, x1_1, x0_2, x3_0, x2_1, x1_2, x0_3)

        if self.config.deepsupervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            if self.training:
                cls_true = y['cls']
                loss_dict = {
                    'cls_loss': self.cls_loss([output1, output2, output3, output4], cls_true)
                }
                return loss_dict
            else:
                return torch.softmax(output4, dim=1)
        else:
            cls_pred = self.final(x0_4)
            if self.training:
                cls_true = y['cls']
                loss_dict = {
                    'cls_loss': self.cls_loss(cls_pred, cls_true)
                }
                return loss_dict
            else:
                return torch.softmax(cls_pred, dim=1)

    def cls_loss(self, y_pred, y_true):
        if self.config.deepsupervision:
            loss = 0
            for idx, pred_i in enumerate(y_pred):
                loss += F.cross_entropy(pred_i, y_true.long(), ignore_index=self.config.loss.ignore_index) *\
                        self.config.loss.loss_weight[idx]
            return loss
        else:
            loss = F.cross_entropy(y_pred, y_true.long(), ignore_index=self.config.loss.ignore_index)

        return loss


    def set_defalut_config(self):
        self.config.update(dict(
            deepsupervision=False,
            num_classes=16,
            cp=(True, True, True, True),
            loss=dict(
                ignore_index=255,
                loss_weight= [1., 1., 1., 1.]
            )
        ))

if __name__ == '__main__':
    # unet = UNet({})
    # unet.eval()
    # x = torch.ones((1, 3, 512, 512))
    # o = unet(x)
    unet = NestedUNet({})
    unet.eval()
    x = torch.ones((1, 3, 512, 512))
    o = unet(x)