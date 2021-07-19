import torch.nn as nn
import torchvision.models as models
from simplecv.module.resnet import ResNetEncoder
from simplecv import registry
from simplecv.interface import CVModule
from module.baseline.base_models.blocks import RefineNetBlock, ResidualConvUnit
import torch.nn.functional as F
import torch
@registry.MODEL.register('RefineNet4Cascade')
class RefineNet4Cascade(CVModule):
    def __init__(self, config):
        super(RefineNet4Cascade, self).__init__(config)
        features = self.config['refine_channels']
        input_size = self.config['input_size']
        self.num_classes = self.config.num_classes
        self.resnet_encoder = ResNetEncoder(self.config.encoder_config)
        self.layer1_rn = nn.Conv2d(
            self.config.channel_list[0], features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(
            self.config.channel_list[1], features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(
            self.config.channel_list[2], features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_rn = nn.Conv2d(
            self.config.channel_list[3], 2 * features, kernel_size=3, stride=1, padding=1, bias=False)

        self.refinenet4 = RefineNetBlock(2 * features,
                                         (2 * features, input_size // 32))
        self.refinenet3 = RefineNetBlock(features,
                                         (2 * features, input_size // 32),
                                         (features, input_size // 16))
        self.refinenet2 = RefineNetBlock(features,
                                         (features, input_size // 16),
                                         (features, input_size // 8))
        self.refinenet1 = RefineNetBlock(features, (features, input_size // 8),
                                         (features, input_size // 4))

        self.output_conv = nn.Sequential(
            ResidualConvUnit(features), ResidualConvUnit(features),
            nn.Conv2d(
                features,
                self.num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True))

    def forward(self, x, y=None):

        # layer_1 = self.layer1(x)
        # layer_2 = self.layer2(layer_1)
        # layer_3 = self.layer3(layer_2)
        # layer_4 = self.layer4(layer_3)
        c2, c3, c4, c5 = self.resnet_encoder(x)

        layer_1_rn = self.layer1_rn(c2)
        layer_2_rn = self.layer2_rn(c3)
        layer_3_rn = self.layer3_rn(c4)
        layer_4_rn = self.layer4_rn(c5)

        path_4 = self.refinenet4(layer_4_rn)
        path_3 = self.refinenet3(path_4, layer_3_rn)
        path_2 = self.refinenet2(path_3, layer_2_rn)
        path_1 = self.refinenet1(path_2, layer_1_rn)
        cls_pred = self.output_conv(path_1)
        cls_pred = F.interpolate(cls_pred, scale_factor=4, mode='bilinear',
                                 align_corners=True)
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
            encoder_config=dict(
                resnet_type='resnet50',
                include_conv5=True,
                batchnorm_trainable=True,
                pretrained=False,
                freeze_at=0,
                # 8, 16 or 32
                output_stride=32,
                with_cp=(False, False, False, False),
                stem3_3x3=False,
            ),
            channel_list = (256, 512, 1024, 2048),
            refine_channels=256,
            num_classes=16,
            input_size=896,
            loss=dict(
                ignore_index=255
            )
        )
        )

if __name__ == '__main__':
    import torch
    x = torch.ones((1, 3, 896, 896))
    m = RefineNet4Cascade({})
    o = m(x)
    print(o)