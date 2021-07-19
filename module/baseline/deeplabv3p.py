from simplecv.interface import CVModule
from simplecv.module.resnet import ResNetEncoder
from simplecv.module import aspp
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from simplecv import registry
from simplecv.module.sep_conv import SeparableConv2D


class LightDecoder(nn.Module):
    """
    This module is a reimplemented version in the following paper.
    Chen L C, Zhu Y, Papandreou G, et al. Encoder-decoder with atrous
    separable convolution for semantic image segmentation[J],
    """

    def __init__(self,
                 os4_feature_channel=256,
                 os16_feature_channel=256,
                 reduction_dim=48,
                 decoder_dim=256,
                 num_3x3convs=2,
                 scale_factor=4.0,
                 bias=True,
                 use_batchnorm=False,
                 ):
        super(LightDecoder, self).__init__()
        self.scale_factor = scale_factor

        self.conv1x1_os4 = nn.Sequential(
            nn.Conv2d(os4_feature_channel, reduction_dim, 1),
            nn.BatchNorm2d(reduction_dim) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        self.conv1x1_os16 = nn.Sequential(
            nn.Conv2d(os16_feature_channel, reduction_dim, 1),
            nn.BatchNorm2d(reduction_dim) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        layers = [SeparableConv2D(reduction_dim * 2, decoder_dim, 3, 1, padding=1, dilation=1,
                                  bias=bias,
                                  norm_fn=nn.BatchNorm2d)] + [
                     SeparableConv2D(decoder_dim, decoder_dim, 3, 1, padding=1, dilation=1,
                                     bias=bias,
                                     norm_fn=nn.BatchNorm2d) for _ in
                     range(num_3x3convs - 1)]
        self.stack_conv3x3 = nn.Sequential(*layers)

    def forward(self, os4_feat, os16_feat):
        low_feat = self.conv1x1_os4(os4_feat)
        encoder_feat = self.conv1x1_os16(os16_feat)

        feat_upx = F.interpolate(encoder_feat, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)

        concat_feat = torch.cat([low_feat, feat_upx], dim=1)

        out = self.stack_conv3x3(concat_feat)

        return out


@registry.MODEL.register('Deeplabv3p')
class Deeplabv3p(CVModule):
    def __init__(self, config):
        super(Deeplabv3p, self).__init__(config)
        self.en = ResNetEncoder(self.config.encoder_config.resnet_encoder)
        self.de = LightDecoder(**self.config.light_decoder)
        self.aspp = aspp.AtrousSpatialPyramidPool(**self.config.aspp)
        self.cls_pred_conv = nn.Conv2d(self.aspp.aspp_dim, self.config.num_classes, 1)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self._reset_bn_momentum(0.1)

    def _reset_bn_momentum(self, momentum):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = momentum

    def forward(self, x, y=None):
        feat_list = self.en(x)
        os4_feat = feat_list[0]
        os16_feat = self.aspp(feat_list[-1])

        feat = self.de(os4_feat, os16_feat)

        cls_pred = self.cls_pred_conv(feat)

        cls_pred = F.interpolate(cls_pred, scale_factor=self.config.upsample_ratio, mode='bilinear',
                                 align_corners=True)

        if self.training:
            cls_true = y['cls']
            loss_dict = {
                'cls_loss': self.config.loss_config.cls_weight * self.cls_loss(cls_pred, cls_true)
            }
            mem = torch.cuda.max_memory_cached() // 1024 // 1024
            loss_dict['mem'] = torch.from_numpy(np.array([mem], dtype=np.float32)).to(self.device)
            return loss_dict

        cls_prob = torch.softmax(cls_pred, dim=1)

        return cls_prob

    def cls_loss(self, y_pred, y_true):
        return F.cross_entropy(y_pred, y_true.long(), ignore_index=self.config.loss.ignore_index)

    def set_defalut_config(self):
        self.config.update(dict(
            encoder_config=dict(
                resnet_encoder=dict(
                    resnet_type='resnet50',
                    include_conv5=True,
                    batchnorm_trainable=True,
                    pretrained=False,
                    freeze_at=0,
                    # 8, 16 or 32
                    output_stride=16,
                    with_cp=(False, False, False, False),
                    stem3_3x3=False,
                ),
            ),
            light_decoder=dict(
                os4_feature_channel=64,
                os16_feature_channel=64,
                reduction_dim=48,
                decoder_dim=64,
                num_3x3convs=2,
                scale_factor=4.0,
                bias=False,
                use_batchnorm=True,
            ),
            aspp=dict(
                in_channel=2048,
                aspp_dim=256,
                atrous_rates=(6, 12, 18),
                add_image_level=True,
                use_bias=False,
                use_batchnorm=True,
                norm_type='batchnorm',
                batchnorm_trainable=True,
            ),
            upsample_ratio=4.0,
            loss_config=dict(
                cls_weight=1.0,
                ignore_index=-1,
            ),
            num_classes=5,
        ))

if __name__ == '__main__':
    deeplabv3p = Deeplabv3p({})
    deeplabv3p.eval()
    from simplecv.util.param_util import count_model_flops, count_model_parameters

    count_model_parameters(deeplabv3p)
    count_model_parameters(deeplabv3p.en)
    count_model_parameters(deeplabv3p.aspp)
    count_model_parameters(deeplabv3p.cls_pred_conv)
    # count_model_flops(deeplabv3p, dict(rgb=torch.ones(1, 3, 512, 512)))