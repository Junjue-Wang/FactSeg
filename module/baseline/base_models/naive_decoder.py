# Date: 2018.10.27
# Author: Kingdrone

import torch.nn as nn
import torch
import torch.utils.checkpoint as cp


class NaiveDecoder(nn.Module):
    def __init__(self,
                 block_dims=(512, 256, 128, 64),
                 max_channel=1024,
                 use_bn=False,
                 use_cp=False,
                 ):
        super().__init__()
        self.module_list = nn.ModuleList()
        self.upsample_list = nn.ModuleList()
        self.use_cp = use_cp
        in_channel = max_channel
        for idx, out_dim in enumerate(block_dims):
            if use_bn:
                block = nn.Sequential(
                    nn.Conv2d(in_channel, out_dim, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(out_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_dim, out_dim, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(out_dim),
                    nn.ReLU(inplace=True)
                )
            else:
                block = nn.Sequential(
                    nn.Conv2d(in_channel, out_dim, 3, 1, 1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_dim, out_dim, 3, 1, 1, bias=False),
                    nn.ReLU(inplace=True))
            self.module_list.append(block)
            upsample_op = nn.Sequential(
                nn.ConvTranspose2d(in_channel, out_dim, 4, 2, 1, bias=False)
            )
            self.upsample_list.append(upsample_op)
            in_channel = out_dim

    def forward(self, x):
        feat_list = x
        assert isinstance(feat_list, list)
        feat_list.reverse()
        x_i = feat_list[0]
        for idx, feat in enumerate(feat_list[: -1]):
            x_i_before = feat_list[idx + 1]
            p_i = self.upsample_list[idx](x_i)
            concat_i = torch.cat([p_i, x_i_before], dim=1)

            if self.use_cp and concat_i.requires_grad:
                out_i = cp.checkpoint(self.module_list[idx], concat_i)
            else:
                out_i = self.module_list[idx](concat_i)
            x_i = out_i
        torch.cuda.empty_cache()
        return x_i
