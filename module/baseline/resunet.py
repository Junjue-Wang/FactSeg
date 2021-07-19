# Date:2019.04.10
# Author: Kingdrone
import torch.nn as nn
import torch
from simplecv.interface import CVModule
from simplecv import registry
from simplecv.module.resnet import ResNetEncoder
from simplecv.module import aspp
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

def conv3x3_bn_relu(in_channels, out_channels, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class LightDecoder(nn.Module):
    def __init__(self, config):
        super(LightDecoder, self).__init__()
        in_channel_list = config['in_channels_list']
        self.use_cp = config['use_cp']
        # os 32
        self.conv1x1_1 = nn.Conv2d(in_channel_list[0], in_channel_list[1], 1)
        # os 16
        self.conv1x1_2 = nn.Conv2d(in_channel_list[1], in_channel_list[2], 1)
        # os 8
        self.conv1x1_3 = nn.Conv2d(in_channel_list[2], in_channel_list[3], 1)

        self.conv3x3_bn_relu_1 = conv3x3_bn_relu(2 * in_channel_list[1], in_channel_list[1])
        self.conv3x3_bn_relu_2 = conv3x3_bn_relu(2 * in_channel_list[2], in_channel_list[2])
        self.conv3x3_bn_relu_3 = conv3x3_bn_relu(2 * in_channel_list[3], in_channel_list[3])

    def _base_forward(self, os32_feat, os16_feat, os8_feat, os4_feat):
        p1 = self.conv1x1_1(os32_feat)
        if os32_feat.shape[-1] != os16_feat.shape[-1]:
            p1 = F.interpolate(p1, scale_factor=2.0)
        f1 = torch.cat([p1, os16_feat], dim=1)
        o1_os16 = self.conv3x3_bn_relu_1(f1)

        c2 = self.conv1x1_2(o1_os16)
        p2 = F.interpolate(c2, scale_factor=2.0)
        f2 = torch.cat([p2, os8_feat], dim=1)
        o2_os8 = self.conv3x3_bn_relu_2(f2)

        c3 = self.conv1x1_3(o2_os8)
        p3 = F.interpolate(c3, scale_factor=2.0)
        f3 = torch.cat([p3, os4_feat], dim=1)
        o3_os4 = self.conv3x3_bn_relu_3(f3)

        return o3_os4

    def forward(self, os32, os16, os8, os4):
        if self.use_cp:
            return checkpoint(self._base_forward, os32, os16, os8, os4)
        else:
            return self._base_forward(os32, os16, os8, os4)


@registry.MODEL.register('ResUnet')
class ResUnet(CVModule):
    def __init__(self, config):
        super(ResUnet, self).__init__(config)

        self.encoder = ResNetEncoder(self.config.encoder_config.resnet_encoder)
        if self.config.encoder_config['aspp_enable']:
            self.aspp = aspp.AtrousSpatialPyramidPool(**self.config.encoder_config['aspp'])

        self.decoder = LightDecoder(self.config.decoder_config['params'])

        self.cls_pred_conv = nn.Conv2d(self.config.decoder_config['cls_in_dim'], self.config.num_classes, 1)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, x, y=None):
        c2, c3, c4, c5 = self.encoder(x)
        if self.config.encoder_config['aspp_enable']:
            c5 = self.aspp(c5)
        os4 = self.decoder(c5, c4, c3, c2)

        cls_pred = self.cls_pred_conv(os4)
        cls_pred = F.interpolate(cls_pred, scale_factor=4.0, mode='bilinear', align_corners=True)
        if self.training:
            cls_true = y['cls']

            loss = self.cls_loss(cls_pred, cls_true)
            loss_dict = {
                'cls_loss': loss
            }
            return loss_dict
        else:
            cls_prob = torch.softmax(cls_pred, dim=1)
            return cls_prob

    def cls_loss(self, y_pred, y_true):
        return F.cross_entropy(y_pred, y_true.long(), ignore_index=self.config.loss.ignore_index)

    def set_defalut_config(self):
        self.config.update(dict(
            encoder_config=dict(
                resnet_encoder=dict(
                    resnet_type='resnet34',
                    include_conv5=True,
                    batchnorm_trainable=True,
                    pretrained=True,
                    freeze_at=0,
                    # 8, 16 or 32
                    output_stride=32,
                    with_cp=(False, False, False, False),
                    stem3_3x3=False,
                ),
                aspp_enable=True,
                aspp=dict(
                    in_channel=512,
                    aspp_dim=256,
                    atrous_rates=(6, 12, 18),
                    add_image_level=True,
                    use_bias=False,
                    use_batchnorm=True,
                    norm_type='batchnorm',
                ),
            ),
            decoder_config=dict(
                type='default',
                params=dict(
                    use_cp=False,
                    in_channels_list=(256, 256, 128, 64),
                ),
                cls_in_dim=64,
            ),
            num_classes=5,
        ))