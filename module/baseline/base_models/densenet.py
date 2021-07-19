import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from simplecv.module.densenet import DenseNetEncoder


class DilatedDenseNetEncoder(DenseNetEncoder):
    def __init__(self, config):
        super(DilatedDenseNetEncoder, self).__init__(config)
        assert (self.config.dilate_scale == 8 or self.config.dilate_scale == 16), "dilate_scale can only set as 8 or 16"
        from functools import partial
        if self.config.dilate_scale == 8:
            self.densenet.features.denseblock3.apply(partial(self._conv_dilate, dilate=2))
            self.densenet.features.denseblock4.apply(partial(self._conv_dilate, dilate=4))
            del self.densenet.features.transition2.pool
            del self.densenet.features.transition3.pool
        elif self.config.dilate_scale == 16:
            self.densenet.features.denseblock4.apply(partial(self._conv_dilate, dilate=2))
            del self.densenet.features.transition3.pool

    def _conv_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.kernel_size == (3, 3):
                m.padding = (dilate, dilate)
                m.dilation = (dilate, dilate)

    def set_defalut_config(self):
        self.config.update(
            dict(
                dilate_scale=8,
                densenet_type='densenet121',
                pretrained=False,
                memory_efficient=False
            )
        )

if __name__ == '__main__':
    img = torch.randn(2, 3, 224, 224)
    model = DilatedDenseNetEncoder({})
    model.eval()
    outputs = model(img)
    print([o.shape for o in outputs])