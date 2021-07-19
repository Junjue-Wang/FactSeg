from module.baseline.base_models.densenet import *
from simplecv.interface import CVModule
import torch.nn.functional as F
import torch
import torch.nn as nn
from simplecv import registry
from module.baseline.base_models.densenet import DilatedDenseNetEncoder


@registry.MODEL.register('DenseASPP')
class DenseASPP(CVModule):
    def __init__(self, config):
        super(DenseASPP, self).__init__(config)
        self.deneseencoder = DilatedDenseNetEncoder(self.config.encoder_config)
        self.head = _DenseASPPHead(self.config.head_inchannels, self.config.num_classes)


    def forward(self, x, y=None):
        size = x.size()[2:]
        c2, c3, c4, c5 = self.deneseencoder(x)

        if self.config.encoder_config.dilate_scale > 8:
            c5 = F.interpolate(c5, scale_factor=2, mode='bilinear', align_corners=True)

        x = self.head(c5)
        cls_pred = F.interpolate(x, size, mode='bilinear', align_corners=True)

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
                dilate_scale=8,
                densenet_type='densenet121',
                pretrained=False,
                memory_efficient=False
            ),
            head_inchannels=1024,
            num_classes = 16,
            loss=dict(
                ignore_index=255
            )
        ))


class _DenseASPPHead(nn.Module):
    def __init__(self, in_channels, nclass, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_DenseASPPHead, self).__init__()
        self.dense_aspp_block = _DenseASPPBlock(in_channels, 256, 64, norm_layer, norm_kwargs)
        self.block = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(in_channels + 5 * 64, nclass, 1)
        )

    def forward(self, x):
        x = self.dense_aspp_block(x)
        return self.block(x)


class _DenseASPPConv(nn.Sequential):
    def __init__(self, in_channels, inter_channels, out_channels, atrous_rate,
                 drop_rate=0.1, norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(_DenseASPPConv, self).__init__()
        self.add_module('conv1', nn.Conv2d(in_channels, inter_channels, 1)),
        self.add_module('bn1', norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs))),
        self.add_module('relu1', nn.ReLU(True)),
        self.add_module('conv2', nn.Conv2d(inter_channels, out_channels, 3, dilation=atrous_rate, padding=atrous_rate)),
        self.add_module('bn2', norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs))),
        self.add_module('relu2', nn.ReLU(True)),
        self.drop_rate = drop_rate

    def forward(self, x):
        features = super(_DenseASPPConv, self).forward(x)
        if self.drop_rate > 0:
            features = F.dropout(features, p=self.drop_rate, training=self.training)
        return features


class _DenseASPPBlock(nn.Module):
    def __init__(self, in_channels, inter_channels1, inter_channels2,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(_DenseASPPBlock, self).__init__()
        self.aspp_3 = _DenseASPPConv(in_channels, inter_channels1, inter_channels2, 3, 0.1,
                                     norm_layer, norm_kwargs)
        self.aspp_6 = _DenseASPPConv(in_channels + inter_channels2 * 1, inter_channels1, inter_channels2, 6, 0.1,
                                     norm_layer, norm_kwargs)
        self.aspp_12 = _DenseASPPConv(in_channels + inter_channels2 * 2, inter_channels1, inter_channels2, 12, 0.1,
                                      norm_layer, norm_kwargs)
        self.aspp_18 = _DenseASPPConv(in_channels + inter_channels2 * 3, inter_channels1, inter_channels2, 18, 0.1,
                                      norm_layer, norm_kwargs)
        self.aspp_24 = _DenseASPPConv(in_channels + inter_channels2 * 4, inter_channels1, inter_channels2, 24, 0.1,
                                      norm_layer, norm_kwargs)

    def forward(self, x):
        aspp3 = self.aspp_3(x)
        x = torch.cat([aspp3, x], dim=1)

        aspp6 = self.aspp_6(x)
        x = torch.cat([aspp6, x], dim=1)

        aspp12 = self.aspp_12(x)
        x = torch.cat([aspp12, x], dim=1)

        aspp18 = self.aspp_18(x)
        x = torch.cat([aspp18, x], dim=1)

        aspp24 = self.aspp_24(x)
        x = torch.cat([aspp24, x], dim=1)

        return x

datasets = {
    'ade20k': 150,
    'pascal_voc': 21,
    'pascal_aug': 21,
    'coco': 21,
    'citys': 19,
    'sbu': 2,
}


# def get_denseaspp(dataset='citys', backbone='densenet121', pretrained=False,
#                   root='~/.torch/models', pretrained_base=False, **kwargs):
#     r"""DenseASPP
#     Parameters
#     ----------
#     dataset : str, default citys
#         The dataset that model pretrained on. (pascal_voc, ade20k)
#     pretrained : bool or str
#         Boolean value controls whether to load the default pretrained weights for model.
#         String value represents the hashtag for a certain version of pretrained weights.
#     root : str, default '~/.torch/models'
#         Location for keeping the model parameters.
#     pretrained_base : bool or str, default True
#         This will load pretrained backbone network, that was trained on ImageNet.
#     Examples
#     --------
#     >>> model = get_denseaspp(dataset='citys', backbone='densenet121', pretrained=False)
#     >>> print(model)
#     """
#     acronyms = {
#         'pascal_voc': 'pascal_voc',
#         'pascal_aug': 'pascal_aug',
#         'ade20k': 'ade',
#         'coco': 'coco',
#         'citys': 'citys',
#     }
#     model = DenseASPP(datasets[dataset], backbone=backbone, pretrained_base=pretrained_base, **kwargs)
#     if pretrained:
#         device = torch.device(kwargs['local_rank'])
#         model.load_state_dict(torch.load(get_model_file('denseaspp_%s_%s' % (backbone, acronyms[dataset]), root=root),
#                               map_location=device))
#     return model
#
# def get_model_file(name, root='~/.torch/models'):
#     root = os.path.expanduser(root)
#     file_path = os.path.join(root, name + '.pth')
#     if os.path.exists(file_path):
#         return file_path
#     else:
#         raise ValueError('Model file is not found. Downloading or trainning.')
#
# def get_denseaspp_densenet121_citys(**kwargs):
#     return get_denseaspp('citys', 'densenet121', **kwargs)
#
#
# def get_denseaspp_densenet161_citys(**kwargs):
#     return get_denseaspp('citys', 'densenet161', **kwargs)
#
#
# def get_denseaspp_densenet169_citys(**kwargs):
#     return get_denseaspp('citys', 'densenet169', **kwargs)
#
#
# def get_denseaspp_densenet201_citys(**kwargs):
#     return get_denseaspp('citys', 'densenet201', **kwargs)

if __name__ == '__main__':
    img = torch.randn(2, 3, 480, 480)
    y = torch.ones(2, 480, 480)
    model = DenseASPP({})
    outputs = model(img, dict(cls=y))
    print(outputs)