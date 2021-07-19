import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
# This is implemented in full accordance with the original one (https://github.com/shelhamer/fcn.berkeleyvision.org)
from simplecv.interface import CVModule
from simplecv import registry
@registry.MODEL.register('FCN8s')
class FCN8s(CVModule):
    def __init__(self, config):
        super(FCN8s, self).__init__(config)
        if self.config.pretrained == True:
            vgg = models.vgg16(pretrained=True)
        else:
            vgg = models.vgg16(pretrained=False)

        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())
        num_classes = self.config.num_classes

        self.features3 = nn.Sequential(*features[: 17])
        self.features4 = nn.Sequential(*features[17: 24])
        self.features5 = nn.Sequential(*features[24:])

        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)

        fc6 = nn.Conv2d(512, 4096, kernel_size=3, stride=1, padding=1)
        fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)

        self.score_fr = nn.Sequential(
            fc6, nn.ReLU(inplace=True), nn.Dropout(), fc7, nn.ReLU(inplace=True), nn.Dropout(), score_fr
        )

        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, padding=4, bias=False)

    def forward(self, x, y=None):
        x_size = x.size()
        pool3 = self.features3(x)   # os8
        pool4 = self.features4(pool3) # os16
        pool5 = self.features5(pool4) # os32

        score_fr = self.score_fr(pool5)
        upscore2 = self.upscore2(score_fr)

        score_pool4 = self.score_pool4(pool4)
        upscore_pool4 = self.upscore_pool4(score_pool4 + upscore2)

        score_pool3 = self.score_pool3(pool3)      # 将通道数标准化为类别数
        cls_pred = self.upscore8(score_pool3 + upscore_pool4)       # 反卷积
        if self.training:
            cls_true = y['cls']
            loss_dict = {
                'cls_loss': self.config.loss.cls_weight * self.cls_loss(cls_pred, cls_true)
            }
            # mem = torch.cuda.max_memory_allocated() // 1024 // 1024
            # loss_dict['mem'] = torch.from_numpy(np.array([mem], dtype=np.float32)).to(self.device)
            return loss_dict

        cls_prob = torch.softmax(cls_pred, dim=1)

        return cls_prob

    def cls_loss(self, y_pred, y_true):
        return F.cross_entropy(y_pred, y_true.long(), ignore_index=self.config.loss.ignore_index)

    def set_defalut_config(self):
        self.config.update(
            dict(
                pretrained=False,
                num_classes=16,
                loss=dict(
                    ignore_index=255,
                    cls_weight=1.0
                )
            )
        )

if __name__ == '__main__':
    fcn = FCN8s({})
    fcn.eval()
    from simplecv.util.param_util import count_model_flops, count_model_parameters
    # count_model_parameters(fcn)
    # count_model_flops(fcn, torch.ones(1, 3, 896, 896))
    o = fcn(torch.ones(1, 3, 896, 896))
    print(o.shape)