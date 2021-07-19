from simplecv.interface import CVModule
from simplecv.module.resnet import ResNetEncoder
from simplecv.module import aspp
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from simplecv import registry

@registry.MODEL.register('Deeplabv3')
class Deeplabv3(CVModule):
    def __init__(self, config):
        super(Deeplabv3, self).__init__(config)
        self.en = ResNetEncoder(self.config.encoder_config.resnet_encoder)
        self.aspp = aspp.AtrousSpatialPyramidPool(**self.config.aspp_config)
        self.cls_pred_conv = nn.Conv2d(self.aspp.aspp_dim, self.config.num_classes, 1)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self._reset_bn_momentum(0.1)
    def _reset_bn_momentum(self, momentum):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = momentum

    def forward(self, x, y=None):
        feat_list = self.en(x)
        c5 = feat_list[-1]

        # feat = self.de(c5)

        logit = self.aspp(c5)
        cls_pred = self.cls_pred_conv(logit)

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
        loss = F.cross_entropy(y_pred, y_true.long(), ignore_index=self.config.loss.ignore_index)
        # if self.config.loss_config['ohem']:
        #     loss = self.sgohem(loss)
        return loss


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
                aspp_config=dict(
                    in_channel=2048,
                    aspp_dim=256,
                    atrous_rates=(6, 12, 18),
                    add_image_level=True,
                    use_bias=False,
                    use_batchnorm=True,
                    norm_type='batchnorm',
                    batchnorm_trainable=True,
                ),
                upsample_ratio=16.0,
                loss_config=dict(
                    cls_weight=1.0,
                    ignore_index=-1,
                ),
                num_classes=16,
            )
        )

if __name__ == '__main__':
    deeplabv3 = Deeplabv3({})
    from simplecv.util.param_util import count_model_flops, count_model_parameters

    deeplabv3.eval()

    # x = dict(rgb=torch.zeros(1, 3, 512, 512))
    # o = deeplabv3(x)
    # print(o.shape)
    count_model_parameters(deeplabv3.en)
    count_model_parameters(deeplabv3.aspp)
    count_model_parameters(deeplabv3)
    # count_model_parameters(deeplabv3.en)
    # count_model_parameters(deeplabv3.aspp)
    # count_model_parameters(deeplabv3.cls_pred_conv)
    # count_model_flops(deeplabv3, dict(rgb=torch.zeros(1, 3, 512, 512)))