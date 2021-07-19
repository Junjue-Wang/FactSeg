# Date: 2018.10.28
import torch
import numpy as np
import torch.nn as nn
from module.baseline.base_models.naive_decoder import NaiveDecoder
from module.baseline.base_models.naive_encoder import NaiveEncoder
from simplecv.interface import CVModule
import torch.nn.functional as F
from simplecv import registry


@registry.MODEL.register('UNet')
class Unet(CVModule):
    def __init__(self,
                 config
                 ):
        super(Unet, self).__init__(config)
        self.naive_encoder = NaiveEncoder(**self.config['encoder'])
        self.naive_decoder = NaiveDecoder(**self.config['decoder'])
        self.num_classes = self.config['num_classes']
        self.cls_pred_conv = nn.Conv2d(self.config['decoder']['block_dims'][-1], self.num_classes, 1, bias=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    def forward(self, x, y=None):
        feat_list = self.naive_encoder(x)
        out = self.naive_decoder(feat_list)
        cls_pred = self.cls_pred_conv(out)
        
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

    @staticmethod
    def get_function(module):
        def _function(x):
            y = module(x)
            return y

        return _function

    def set_defalut_config(self):
        self.config.update(dict(
            encoder=dict(
                block_dims=(32, 64, 128, 256, 512),
                use_bn=True,
                use_cp=False,
            ),
            decoder=dict(
                block_dims=(256, 128, 64, 32),
                max_channel=512,
                use_bn=True,
                use_cp=False,
            ),
            num_classes=15,
        )
        )


if __name__ == '__main__':
    unet = Unet({})
    unet.eval()
    from simplecv.util.param_util import count_model_parameters
    count_model_parameters(Unet({}))

