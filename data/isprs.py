from data.patch_base import PatchBasedDataset
from data.patch_base import DEFAULT_PATCH_CONFIG
from simplecv.util import viz
import matplotlib.pyplot as plt
import os
from simplecv.api.preprocess import segm, comm
from simplecv.data import distributed
from torch.utils.data import SequentialSampler
from simplecv import registry
from simplecv.core.config import AttrDict
from torch.utils.data.dataloader import DataLoader
from PIL import Image
from collections import OrderedDict
import numpy as np
import torch
from functools import reduce

COLOR_MAP = dict(
    Vaihingen=OrderedDict(
        ImpSurf=(255, 255, 255),
        Building=(0, 0, 255),
        LowVeg=(0, 255, 255),
        Tree=(0, 255, 0),
        Car=(255, 255, 0)
    ),
    Potsdam=OrderedDict(
        ImpSurf=(255, 255, 255),
        Building=(0, 0, 255),
        LowVeg=(0, 255, 255),
        Tree=(0, 255, 0),
        Car=(255, 255, 0),
        Clutter=(255, 0, 0)
    )
)


class ISPRSDataset(PatchBasedDataset):
    DATA_SPLIT = dict(
        Potsdam=dict(
            train=['2_10', '3_10', '3_11', '3_12', '4_11', '4_12', '5_10', '5_12', '6_10', '6_11', '6_12', '6_8', '6_9',
                   '7_11',
                   '7_12', '7_7', '7_9'],
            val=['2_11', '2_12', '4_10', '5_11', '6_7', '7_10', '7_8'],
            label_template='top_potsdam_{id}_label.tif',
            label_noBoundary_template='top_potsdam_{id}_label_noBoundary.tif',
            image_template='top_potsdam_{id}_RGB.tif'),
        Vaihingen=dict(
            train=[13, 17, 1, 21, 23, 26, 32, 37, 3, 5, 7],
            val=[11, 15, 28, 30, 34],
            label_template='top_mosaic_09cm_area{id}.tif',
            label_noBoundary_template='top_mosaic_09cm_area{id}_noBoundary.tif',
            image_template='top_mosaic_09cm_area{id}.tif'))

    def __init__(self, image_dir, mask_dir, type, patch_config=DEFAULT_PATCH_CONFIG, transforms=None, cls_classes=None):
        """
        Args:
            image_dir:
            mask_dir:
            type: ('Potsdam', 'train' or 'val', 'noBoundary') or ('Vaihingen', 'train' or 'val', 'noBoundary')
        """

        self.data_type = type
        self.cls_classes = cls_classes
        super(ISPRSDataset, self).__init__(image_dir, mask_dir, patch_config, transforms)

    def generate_path_pair(self):
        info = ISPRSDataset.DATA_SPLIT[self.data_type[0]]

        image_names = [info['image_template'].format(id=id) for id in info[self.data_type[1]]]
        image_path = [os.path.join(self.image_dir, imname) for imname in image_names]
        if len(self.data_type) > 2 and self.data_type[2] == 'noBoundary':
            label_names = [info['label_noBoundary_template'].format(id=id) for id in info[self.data_type[1]]]
        else:
            label_names = [info['label_template'].format(id=id) for id in info[self.data_type[1]]]
        label_path = [os.path.join(self.mask_dir, lbname) for lbname in label_names]

        path_pair = [(impath, lbpath) for impath, lbpath in zip(image_path, label_path)]
        if self.data_type[1] == 'val':
            self._val_path_pair = path_pair
        return path_pair

    def __getitem__(self, idx):
        if self.data_type[1] == 'val':
            impath, maskpath = self._val_path_pair[idx]
            image = Image.open(impath)
            mask = Image.open(maskpath)
            if self.transforms.transforms is not None:
                image, mask = self.transforms(image, mask)

            if self.cls_classes is not None:
                mask = self.reclassify(mask)
            # mask = torch.where(reduce(lambda a, b: a & b, [mask != v for v in [0, 1, 4]]), torch.zeros_like(mask), mask)
            # mask = torch.where(mask == 4, torch.ones_like(mask) * 2, mask)

            return image, mask, os.path.basename(impath)

        image_path, mask_path, win = self._data_list[idx]

        image = Image.open(image_path)
        mask = Image.open(mask_path)

        image = image.crop(win)
        mask = mask.crop(win)

        if self.transforms.transforms is not None:
            image, mask = self.transforms(image, mask)
        # 1 building, 4 car
        # 0 background, 1 building, 2 car
        # mask = torch.where(reduce(lambda a, b: a & b, [mask != v for v in [0, 1, 4]]), torch.zeros_like(mask), mask)
        # mask = torch.where(mask == 4, torch.ones_like(mask) * 2, mask)
        if self.cls_classes is not None:
            mask = self.reclassify(mask)
        return image, dict(cls=mask)

    def reclassify(self, mask):
        # ImpSurf - 0, Building - 1, LowVeg - 2, Tree - 3, Car - 4
        mask = torch.where(reduce(lambda a, b: a & b, [mask != v for v in self.cls_classes]), torch.zeros_like(mask), mask)
        for idx, v in enumerate(self.cls_classes):
            mask = torch.where(mask == v, torch.ones_like(mask) * (idx + 1), mask)
        return mask

    def __len__(self):
        if self.data_type[1] == 'val':
            return len(self._val_path_pair)
        return len(self._data_list)


@registry.DATALOADER.register('ISPRSDataLoader')
class ISPRSDataLoader(DataLoader):
    def __init__(self, config):
        self.config = AttrDict()
        self.set_defalut()
        self.config.update(config)

        dataset = ISPRSDataset(self.config.image_dir,
                               self.config.mask_dir,
                               self.config.type,
                               self.config.patch_config,
                               self.config.transforms,
                               self.config.cls_classes
                               )
        print(self.config.type)
        sampler = distributed.StepDistributedSampler(dataset) if self.config.training else SequentialSampler(
            dataset)

        super(ISPRSDataLoader, self).__init__(dataset,
                                              self.config.batch_size,
                                              sampler=sampler,
                                              num_workers=self.config.num_workers,
                                              pin_memory=True,
                                              drop_last=True)

    def set_defalut(self):
        self.config.update(dict(
            cls_classes=None,
            image_dir='',
            mask_dir='',
            type=('', 'train'),
            patch_config=dict(
                patch_size=512,
                stride=256,
            ),
            transforms=[
                segm.RandomHorizontalFlip(0.5),
                segm.RandomVerticalFlip(0.5),
                segm.FixedPad((512, 512), 255),
                segm.ToTensor(True),
                comm.THMeanStdNormalize((123.675, 116.28, 103.53), (58.395, 57.12, 57.375))
            ],
            batch_size=1,
            num_workers=0,
            training=True
        ))


if __name__ == '__main__':
    dataset = ISPRSDataset(r'D:\DATA\isprs\exp_setting\Potsdam\images',
                           r'D:\DATA\isprs\exp_setting\Potsdam\nocolor_masks',
                           type=('Potsdam', 'train'),
                           transforms=[
                               segm.ToTensor(True),
                           ])
    print(len(dataset))
