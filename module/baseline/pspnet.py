import torch
from torch import nn
from torch.nn import functional as F
from simplecv.module.resnet import ResNetEncoder
from simplecv.interface import CVModule
from simplecv import registry
import torch.utils.checkpoint as cp


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.interpolate(input=x, size=(h, w), mode='bilinear', align_corners=True)
        return self.conv(p)

@registry.MODEL.register('PSPNet')
class PSPNet(CVModule):
    def __init__(self, config):
        super().__init__(config)

        self.encoder = ResNetEncoder(self.config.encoder_config.resnet_encoder)
        self.psp = PSPModule(self.config.psp_size, 1024, self.config.sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.cls_pred_conv = nn.Conv2d(64, self.config.num_classes, kernel_size=1)

    def forward(self, x, y=None):
        c2, c3, c4, c5 = self.encoder(x)

        p = self.psp(c5)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)
        cls_pred = self.cls_pred_conv(p)

        if self.training:
            cls_true = y['cls']
            loss_dict = {
                'cls_loss': self.config.loss_config.cls_weight * self.cls_loss(cls_pred, cls_true)
            }
            # mem = torch.cuda.max_memory_allocated() // 1024 // 1024
            # loss_dict['mem'] = torch.from_numpy(np.array([mem], dtype=np.float32)).to(self.device)
            return loss_dict

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
                    pretrained=False,
                    freeze_at=0,
                    # 8, 16 or 32
                    output_stride=8,
                    with_cp=(False, False, False, False),
                    stem3_3x3=False,
                )
            ),
            num_classes=5,
            sizes=(1, 2, 3, 6),
            psp_size=512,
            loss_config=dict(
                cls_weight=1.0,
                ignore_index=-1,
            ),
        ))

if __name__ == '__main__':
    # fcn = PSPNet({})
    # fcn.eval()
    from simplecv.util.param_util import count_model_flops, count_model_parameters
    #
    # count_model_parameters(fcn)
    # count_model_flops(fcn, dict(rgb=torch.ones(1, 3, 512, 512)))
    x = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, groups=64))
    y = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3))
    count_model_flops(x, torch.ones(1, 512, 512, 512))
    count_model_flops(y, torch.ones(1, 512, 512, 512))
