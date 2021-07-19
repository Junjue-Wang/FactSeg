import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from simplecv.interface import CVModule
from simplecv import registry
from simplecv.module import resnet
from simplecv.module import fpn
import math
from simplecv.module.resnet import plugin_context_block2d
from module.loss import softmax_focalloss
from module.loss import cosineannealing_softmax_focalloss

try:
    from module.dcn import resnet_plugin
except:
    pass


class AssymetricDecoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_feat_output_strides=(4, 8, 16, 32),
                 out_feat_output_stride=4,
                 norm_fn=nn.BatchNorm2d,
                 num_groups_gn=None):
        super(AssymetricDecoder, self).__init__()
        if norm_fn == nn.BatchNorm2d:
            norm_fn_args = dict(num_features=out_channels)
        elif norm_fn == nn.GroupNorm:
            if num_groups_gn is None:
                raise ValueError('When norm_fn is nn.GroupNorm, num_groups_gn is needed.')
            norm_fn_args = dict(num_groups=num_groups_gn, num_channels=out_channels)
        else:
            raise ValueError('Type of {} is not support.'.format(type(norm_fn)))
        self.blocks = nn.ModuleList()
        for in_feat_os in in_feat_output_strides:
            num_upsample = int(math.log2(int(in_feat_os))) - int(math.log2(int(out_feat_output_stride)))

            num_layers = num_upsample if num_upsample != 0 else 1

            self.blocks.append(nn.Sequential(*[
                nn.Sequential(
                    nn.Conv2d(in_channels if idx == 0 else out_channels, out_channels, 3, 1, 1, bias=False),
                    norm_fn(**norm_fn_args) if norm_fn is not None else nn.Identity(),
                    nn.ReLU(inplace=True),
                    nn.UpsamplingBilinear2d(scale_factor=2) if num_upsample != 0 else nn.Identity(),
                )
                for idx in range(num_layers)]))

    def forward(self, feat_list: list):
        inner_feat_list = []
        for idx, block in enumerate(self.blocks):
            decoder_feat = block(feat_list[idx])
            inner_feat_list.append(decoder_feat)

        out_feat = sum(inner_feat_list) / 4.
        return out_feat

@registry.MODEL.register('SemanticFPN')
class SemanticFPN(CVModule):
    def __init__(self, config):
        super(SemanticFPN, self).__init__(config)
        self.register_buffer('buffer_step', torch.zeros((), dtype=torch.float32))

        self.en = resnet.ResNetEncoder(self.config.resnet_encoder)
        self.fpn = fpn.FPN(**self.config.fpn)
        self.decoder = AssymetricDecoder(**self.config.decoder)
        self.cls_pred_conv = nn.Conv2d(self.config.decoder.out_channels, self.config.num_classes, 1)
        self.upsample4x_op = nn.UpsamplingBilinear2d(scale_factor=4)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        if self.config.resnet_encoder.gc_blocks.on:
            self.en.resnet.layer2 = plugin_context_block2d(self.en.resnet.layer2,
                                                           self.config.resnet_encoder.gc_blocks.ratios[0])
            self.en.resnet.layer3 = plugin_context_block2d(self.en.resnet.layer3,
                                                           self.config.resnet_encoder.gc_blocks.ratios[1])
            self.en.resnet.layer4 = plugin_context_block2d(self.en.resnet.layer4,
                                                           self.config.resnet_encoder.gc_blocks.ratios[2])

        if self.config.resnet_encoder.with_dcn[0]:
            self.en.layer1 = resnet_plugin.plugin_dcn(self.en.layer1, self.config.resnet_encoder.dcn)
        if self.config.resnet_encoder.with_dcn[1]:
            self.en.layer2 = resnet_plugin.plugin_dcn(self.en.layer2, self.config.resnet_encoder.dcn)
        if self.config.resnet_encoder.with_dcn[2]:
            self.en.layer3 = resnet_plugin.plugin_dcn(self.en.layer3, self.config.resnet_encoder.dcn)
        if self.config.resnet_encoder.with_dcn[3]:
            self.en.layer4 = resnet_plugin.plugin_dcn(self.en.layer4, self.config.resnet_encoder.dcn)

    def forward(self, x, y=None, return_feature=False):
        feat_list = self.en(x)
        fpn_feat_list = self.fpn(feat_list)
        final_feat = self.decoder(fpn_feat_list)
        cls_pred = self.cls_pred_conv(final_feat)
        cls_pred = self.upsample4x_op(cls_pred)

        feature_pack = dict(backbone=feat_list, neck=fpn_feat_list)
        if self.training:
            cls_true = y['cls']
            loss_dict = dict()
            self.buffer_step += 1
            cls_loss_v = self.config.loss.cls_weight * self.cls_loss(cls_pred, cls_true)
            loss_dict['cls_loss'] = cls_loss_v

            mem = torch.cuda.max_memory_allocated() // 1024 // 1024
            loss_dict['mem'] = torch.from_numpy(np.array([mem], dtype=np.float32)).to(self.device)
            if return_feature:
                return feature_pack, loss_dict
            return loss_dict

        if return_feature:
            return feature_pack, cls_pred.softmax(dim=1)
        return cls_pred.softmax(dim=1)

    def cls_loss(self, y_pred, y_true):
        if 'softmax_focalloss' in self.config:
            return softmax_focalloss(y_pred, y_true.long(), ignore_index=self.config.loss.ignore_index,
                                     gamma=self.config.softmax_focalloss.gamma, normalize=self.config.normalize)
        elif 'cosineannealing_softmax_focalloss' in self.config:
            return cosineannealing_softmax_focalloss(y_pred, y_true.long(),
                                                     self.buffer_step.item(),
                                                     self.config.cosineannealing_softmax_focalloss.max_step,
                                                     self.config.loss.ignore_index,
                                                     self.config.cosineannealing_softmax_focalloss.gamma)
        return F.cross_entropy(y_pred, y_true.long(), ignore_index=self.config.loss.ignore_index)

    def set_defalut_config(self):
        self.config.update(dict(
            resnet_encoder=dict(
                resnet_type='resnet50',
                include_conv5=True,
                batchnorm_trainable=True,
                pretrained=False,
                freeze_at=0,
                # 8, 16 or 32
                output_stride=32,
                with_cp=(False, False, False, False),
                stem3_3x3=False,
                gc_blocks=dict(
                    on=False,
                    ratios=(1 / 16., 1 / 16., 1 / 16.)
                ),
                norm_layer=nn.BatchNorm2d,
                with_dcn=(False, False, False, False),
                dcn=dict(fallback_on_stride=False,
                         modulated=False,
                         deformable_groups=1)
            ),
            fpn=dict(
                in_channels_list=(256, 512, 1024, 2048),
                out_channels=256,
                conv_block=fpn.default_conv_block,
                top_blocks=None,
            ),
            decoder=dict(
                in_channels=256,
                out_channels=128,
                in_feat_output_strides=(4, 8, 16, 32),
                out_feat_output_stride=4,
                norm_fn=nn.BatchNorm2d,
                num_groups_gn=None
            ),
            num_classes=16,
            loss=dict(
                cls_weight=1.0,
                ignore_index=255,
            )
        ))