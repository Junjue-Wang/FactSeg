# Date: 2018.10.26

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

class NaiveEncoder(nn.Module):
    def __init__(self,
                 block_dims=(64, 128, 256, 512, 1024),
                 use_bn=False,
                 use_cp=False,
                 ):
        super().__init__()
        inchannels = 3
        self.maxpool2d = nn.MaxPool2d(2)
        self.module_list = nn.ModuleList()
        self.use_cp = use_cp
        for idx, out_dim in enumerate(block_dims):
            if use_bn:
                block = nn.Sequential(
                    nn.Conv2d(inchannels, out_dim, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(out_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_dim, out_dim, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(out_dim),
                    nn.ReLU(inplace=True)
                )
            else:
                block = nn.Sequential(
                    nn.Conv2d(inchannels, out_dim, 3, 1, 1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_dim, out_dim, 3, 1, 1, bias=False),
                    nn.ReLU(inplace=True)
                )
            inchannels = out_dim
            self.module_list.append(block)

    def forward(self, x):
        feat_list = []
        for block in self.module_list:
            if self.use_cp and x.requires_grad:
                x = cp.checkpoint(block, x)
            else:
                x = block(x)
            feat_list.append(x)
            x = self.maxpool2d(x)
            torch.cuda.empty_cache()
        return feat_list