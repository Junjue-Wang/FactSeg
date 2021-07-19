# Date: 2019.09.07
# Author: Kingdrone

from simplecv.module.resnet import ResNetEncoder
from simplecv.module.fpn import FPN
from module.semantic_fpn import AssymetricDecoder
from simplecv.interface import CVModule
from simplecv import registry
import torch
import torch.nn as nn
import torch.nn.functional as F
from module.loss import JointLoss
from module.loss import ohem_cross_entropy, InverseWeightCrossEntroyLoss

def som(loss, ratio):
    # 1. keep num
    num_inst = loss.numel()
    num_hns = int(ratio * num_inst)
    # 2. select loss
    top_loss, _ = loss.reshape(-1).topk(num_hns, -1)
    loss_mask = (top_loss != 0)
    # 3. mean loss

    return torch.sum(top_loss[loss_mask]) / (loss_mask.sum())


@registry.MODEL.register('FactSeg')
class FactSeg(CVModule):
    def __init__(self, config):
        super(FactSeg, self).__init__(config)
        self.resencoder = ResNetEncoder(self.config.resnet_encoder)
        # encoder attention
        print('use fpn!')
        self.fgfpn = FPN(**self.config.foreground.fpn)
        self.bifpn = FPN(**self.config.binary.fpn)
        # decoder
        
        self.fg_decoder = AssymetricDecoder(**self.config.foreground.assymetric_decoder)
      
        self.bi_decoder = AssymetricDecoder(**self.config.binary.assymetric_decoder)
        
        self.fg_cls = nn.Conv2d(self.config.foreground.out_channels, self.config.num_classes, kernel_size=1)
        self.bi_cls = nn.Conv2d(self.config.binary.out_channels, 1, kernel_size=1)

        # loss
        if 'joint_loss' in self.config.loss:
            self.joint_loss = JointLoss(**self.config.loss.joint_loss)
        if 'inverse_ce' in self.config.loss:
            self.inversece_loss = InverseWeightCrossEntroyLoss(self.config.num_classes, 255)

    def forward(self, x, y=None):
        print(x.shape)
        #x, cls_true = x[:, :3, :, :], x[:, -1, :, :]
    
        feat_list = self.resencoder(x)
        if 'skip_decoder' in self.config.foreground:
            fg_out = self.fgskip_deocder(feat_list)
            bi_out = self.bgskip_deocder(feat_list)
        else:
            forefeat_list = list(self.fgfpn(feat_list))
            binaryfeat_list = self.bifpn(feat_list)

            if self.config.fbattention.atttention:
                for i in range(len(binaryfeat_list)):
                    forefeat_list[i] = self.fbatt_block_list[i](binaryfeat_list[i], forefeat_list[i])

            fg_out = self.fg_decoder(forefeat_list)
            bi_out = self.bi_decoder(binaryfeat_list)

        fg_pred = self.fg_cls(fg_out)
        bi_pred = self.bi_cls(bi_out)
        fg_pred = F.interpolate(fg_pred, scale_factor=4.0, mode='bilinear',
                                 align_corners=True)
        bi_pred = F.interpolate(bi_pred, scale_factor=4.0, mode='bilinear',
                                align_corners=True)
        if self.training:
            cls_true = y['cls']
            if 'joint_loss' in self.config.loss:
                return dict(joint_loss =self.joint_loss(fg_pred, bi_pred, cls_true))
            else:
                return self.cls_loss(fg_pred, bi_pred.squeeze(dim=1), cls_true)
        
        else:
            if 'joint_loss' in self.config.loss:
                binary_prob = torch.sigmoid(bi_pred)
                cls_prob = torch.softmax(fg_pred, dim=1)
                cls_prob[:, 0, :, :] = cls_prob[:, 0, :, :] * (1- binary_prob).squeeze(dim=1)
                cls_prob[:, 1:, :, :] = cls_prob[:, 1:, :, :] * binary_prob
                Z = torch.sum(cls_prob, dim=1)
                cls_prob = cls_prob.div_(Z)
                return cls_prob
            else:
                return torch.softmax(fg_pred, dim=1)

        
        #else:
        #     binary_prob = torch.sigmoid(bi_pred)
        #     cls_prob = torch.softmax(fg_pred, dim=1)
        #     cls_prob[:, 0, :, :] = cls_prob[:, 0, :, :] * (1- binary_prob).squeeze(dim=1)
        #     cls_prob[:, 1:, :, :] = cls_prob[:, 1:, :, :] * binary_prob
        #     Z = torch.sum(cls_prob, dim=1)
        #     cls_prob = cls_prob.div_(Z)
        #     losses = F.nll_loss(torch.log(cls_prob), cls_true.long(), ignore_index=255, reduction='none')
             
        #     return torch.cat([binary_prob, losses[:, None, :, :]], dim=1)


    def cls_loss(self, fg_pred, bi_pred, cls_true):
        valid_mask = cls_true != 255
        binary_true = torch.where(cls_true>0, torch.ones_like(cls_true), torch.zeros_like(cls_true))
        bce_loss = F.binary_cross_entropy_with_logits(bi_pred[valid_mask].float(), binary_true[valid_mask].float(), reduction='none')

        if 'ohem' in self.config.loss:
            cls_loss = ohem_cross_entropy(fg_pred, cls_true.long(), ignore_index=255)
            return dict(ohem_loss=cls_loss, bce_loss=bce_loss.mean())
        if 'inverse_ce' in self.config.loss:
            cls_loss = self.inversece_loss(fg_pred, cls_true.long())
            return dict(inverse_celoss=cls_loss, bce_loss=bce_loss.mean())
        cls_loss = F.cross_entropy(fg_pred, cls_true.long(), reduction='none', ignore_index=255)
        if 'som' in self.config.loss:
            cls_loss = som(cls_loss, self.config.loss.som)
            #bce_loss = som(bce_loss, self.config.loss.som)
            return dict(som_cls_loss=cls_loss, bce_loss=bce_loss.mean())
        return dict(cls_loss=cls_loss, bce_loss=bce_loss.mean())

    def set_defalut_config(self):
        self.config.update(dict(
            resnet_encoder=dict(
                resnet_type='resnet50',
                include_conv5=True,
                batchnorm_trainable=True,
                pretrained=True,
                freeze_at=0,
                # 8, 16 or 32
                output_stride=32,
                with_cp=(False, False, False, False),
                stem3_3x3=False,
            ),
            num_classes=16,
            foreground=dict(
                fpn=dict(
                    in_channels_list=[256, 512, 1024, 2048],
                    out_channels=256,
                ),
                assymetric_decoder=dict(
                    in_channels=256,
                    out_channels=128,
                    in_feat_output_strides=(4, 8, 16, 32),
                    out_feat_output_stride=4,
                ),
                out_channels=128,
            ),
            binary = dict(
                fpn=dict(
                    in_channels_list=[256, 512, 1024, 2048],
                    out_channels=256,
                ),
                out_channels=128,
                assymetric_decoder=dict(
                    in_channels=256,
                    out_channels=128,
                    in_feat_output_strides=(4, 8, 16, 32),
                    out_feat_output_stride=4,
                ),
            ),
            loss=dict(
            
                ignore_index=255,
                
            )
        ))

