import segmentation_models_pytorch as smp
import ever as er
import torch.nn.functional as F
from module.loss import FocalLoss
import numpy as np
from simplecv.interface import CVModule
from simplecv import registry
@registry.MODEL.register('AnyUNet')
class AnyUNet(CVModule):
    def __init__(self, config):
        super(AnyUNet, self).__init__(config)
        self.features = smp.Unet(self.config.encoder_name,
                                 encoder_weights=self.config.encoder_weights,
                                 classes=self.config.num_classes)
        self.focal_loss = FocalLoss(**self.config.focal_loss)
    def forward(self, x, y=None):
        logit = self.features(x)

        if self.training:
            return dict(cls_loss=self.focal_loss(logit, y['cls'].long()))

        return logit.softmax(dim=1)



    def set_defalut_config(self):
        self.config.update(dict(
            encoder_name='densenet121',
            classes=16,
            encoder_weights=None,
            focal_loss=dict(
                alpha=np.ones(16),
                gamma=2.0,
                ignore_index=255,
                reduction=True
            )
        ))

if __name__ == '__main__':
    import torch
    unet = AnyUNet({})
    x = torch.ones((2, 3, 512, 512))
    y = torch.ones((2, 512, 512))
    # unet.eval()
    o = unet(x, dict(cls=y))
    print(o)