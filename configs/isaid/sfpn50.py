import torch.nn as nn
from simplecv.module import fpn

from data.isaid import RemoveColorMap
from simplecv.api.preprocess import Pipeline
from simplecv.api.preprocess import THMeanStdNormalize2
from simplecv.api.preprocess import THDivisiblePad
from simplecv.api.preprocess import THChannelFirst2
from simplecv.api.preprocess import segm

config = dict(
    model=dict(
        type='SemanticFPN',
        params=dict(
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
        )
    ),
    data=dict(
        train=dict(
            type='ISAIDSegmmDataLoader',
            params=dict(
                image_dir='./isaid_segm/train/images',
                mask_dir='./isaid_segm/train/masks',
                patch_config=dict(
                    patch_size=896,
                    stride=512,
                ),
                transforms=Pipeline(
                    RemoveColorMap(),
                    segm.THRandomVerticalFlip(0.5),
                    segm.THRandomHorizontalFlip(0.5),
                    segm.THRandomRotate90k(),
                    THChannelFirst2(),
                    THMeanStdNormalize2((123.675, 116.28, 103.53), (58.395, 57.12, 57.375)),
                    THDivisiblePad(32, 255, (896, 896)),
                ),
                batch_size=4,
                num_workers=2,
                training=True
            ),
        ),
        test=dict(
            type='ISAIDSegmmDataLoader',
            params=dict(
                image_dir='./isaid_segm/val/images',
                mask_dir='./isaid_segm/val/masks',
                patch_config=dict(
                    patch_size=896,
                    stride=512,
                ),
                transforms=Pipeline(
                    RemoveColorMap(),
                    THChannelFirst2(),
                    THMeanStdNormalize2((123.675, 116.28, 103.53), (58.395, 57.12, 57.375)),
                ),
                batch_size=1,
                num_workers=2,
                training=False
            ),
        ),
    ),
    optimizer=dict(
        type='sgd',
        params=dict(
            momentum=0.9,
            weight_decay=0.0001
        ),
        grad_clip=dict(
            max_norm=35,
            norm_type=2,
        )
    ),
    learning_rate=dict(
        type='poly',
        params=dict(
            base_lr=0.007,
            power=0.9,
            max_iters=60000,
        )),
    train=dict(
        forward_times=1,
        num_iters=60000,
        eval_per_epoch=False,
        summary_grads=False,
        summary_weights=False,
        distributed=True,
        apex_sync_bn=True,
        sync_bn=False,
        eval_after_train=True,
        log_interval_step=50,
    ),
    test=dict(
    ),
)
