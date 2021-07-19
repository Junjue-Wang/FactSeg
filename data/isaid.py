from data.patch_base import PatchBasedDataset
import glob
import os
from collections import OrderedDict
import torch
import torch.nn.functional as F
from simplecv.util import viz
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from skimage.io import imread, imsave
from simplecv.api.preprocess import comm
from simplecv.api.preprocess import segm
from simplecv import registry
from simplecv.core.config import AttrDict
from simplecv.data import distributed
from torch.utils.data import SequentialSampler
import numpy as np
from PIL import Image

DEFAULT_PATCH_CONFIG = dict(
    patch_size=896,
    stride=512,
)
COLOR_MAP = OrderedDict({
    'background': (0, 0, 0),
    'ship': (0, 0, 63),
    'storage_tank': (0, 191, 127),
    'baseball_diamond': (0, 63, 0),
    'tennis_court': (0, 63, 127),
    'basketball_court': (0, 63, 191),
    'ground_Track_Field': (0, 63, 255),
    'bridge': (0, 127, 63),
    'large_Vehicle': (0, 127, 127),
    'small_Vehicle': (0, 0, 127),
    'helicopter': (0, 0, 191),
    'swimming_pool': (0, 0, 255),
    'roundabout': (0, 63, 63),
    'soccer_ball_field': (0, 127, 191),
    'plane': (0, 127, 255),
    'harbor': (0, 100, 155),
})


class RemoveColorMap(object):
    def __init__(self, color_map=COLOR_MAP, mapping=(1, 2, 3)):
        super(RemoveColorMap, self).__init__()
        self.mapping_mat = np.array(mapping).reshape((3, 1))
        features = np.asarray(list(color_map.values()))
        self.keys = np.matmul(features, self.mapping_mat).squeeze()
        self.labels = np.arange(features.shape[0])

    def __call__(self, image, mask):
        if isinstance(mask, Image.Image):
            mask = np.array(mask, copy=False)

        q = np.matmul(mask, self.mapping_mat).squeeze()

        # loop for each class
        out = np.zeros_like(q)
        for label, k in zip(self.labels, self.keys):
            out += np.where(q == k, label * np.ones_like(q), np.zeros_like(q))

        return image, Image.fromarray(out.astype(np.uint8, copy=False))


class ISAIDSegmmDataset(PatchBasedDataset):
    def __init__(self,
                 image_dir,
                 mask_dir,
                 patch_config=DEFAULT_PATCH_CONFIG,
                 transforms=None):
        super(ISAIDSegmmDataset, self).__init__(image_dir, mask_dir, patch_config, transforms=transforms)

    def generate_path_pair(self):
        image_path_list = glob.glob(os.path.join(self.image_dir, '*.png'))

        mask_path_list = [os.path.join(self.mask_dir, os.path.basename(imfp).replace('.png',
                                                                                     '_instance_color_RGB.png')) for
                          imfp in image_path_list]

        return zip(image_path_list, mask_path_list)

    def show_image_mask(self, idx, mask_on=True, ax=None):
        img_tensor, blob = self[idx]
        img = img_tensor.numpy()
        mask = blob['cls'].numpy()
        if mask_on:
            img = np.where(mask.sum() == 0, img, img * 0.5 + (1 - 0.5) * mask)

        viz.plot_image(img, ax)

    def __getitem__(self, idx):
        img_tensor, y = super(ISAIDSegmmDataset, self).__getitem__(idx)
        mask_tensor = y['cls']
        # rm background
        multi_cls_label = torch.unique(mask_tensor)
        # start from 0
        fg_cls_label = multi_cls_label[(multi_cls_label > 0) & (multi_cls_label != 255)] - 1
        y['fg_cls_label'] = F.one_hot(fg_cls_label.long(), num_classes=len(COLOR_MAP) - 1).sum(dim=0)
        return img_tensor, y


@registry.DATALOADER.register('ISAIDSegmmDataLoader')
class ISAIDSegmmDataLoader(DataLoader):
    def __init__(self, config):
        self.config = AttrDict()
        self.set_defalut()
        self.config.update(config)

        dataset = ISAIDSegmmDataset(self.config.image_dir,
                                    self.config.mask_dir,
                                    self.config.patch_config,
                                    self.config.transforms)

        sampler = distributed.StepDistributedSampler(dataset) if self.config.training else SequentialSampler(
            dataset)

        super(ISAIDSegmmDataLoader, self).__init__(dataset,
                                                   self.config.batch_size,
                                                   sampler=sampler,
                                                   num_workers=self.config.num_workers,
                                                   pin_memory=True)

    def set_defalut(self):
        self.config.update(dict(
            image_dir='',
            mask_dir='',
            patch_config=dict(
                patch_size=896,
                stride=512,
            ),
            transforms=[
                RemoveColorMap(),
                segm.RandomHorizontalFlip(0.5),
                segm.RandomVerticalFlip(0.5),
                segm.RandomRotate90K((0, 1, 2, 3)),
                segm.FixedPad((896, 896), 255),
                segm.ToTensor(True),
                comm.THMeanStdNormalize((123.675, 116.28, 103.53), (58.395, 57.12, 57.375))
            ],
            batch_size=1,
            num_workers=0,
            training=True
        ))


class ImageFolderDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None):
        self.fp_list = glob.glob(os.path.join(image_dir, '*.png'))
        self.mask_dir = mask_dir
        self.rm_color = RemoveColorMap()

    def __getitem__(self, idx):
        image_np = imread(self.fp_list[idx])
        if self.mask_dir is not None:
            mask_fp = os.path.join(self.mask_dir, os.path.basename(self.fp_list[idx]).replace('.png',
                                                                                              '_instance_color_RGB.png'))
            mask_np = imread(mask_fp)
            _, mask = self.rm_color(None, mask_np)
            mask_np = np.array(mask, copy=False)
        else:
            mask_np = None
        if len(image_np.shape) == 2:
            image_np = np.stack([image_np] * 3, axis=2)
        return image_np, mask_np, os.path.basename(self.fp_list[idx])

    def __len__(self):
        return len(self.fp_list)

if __name__ == '__main__':
    from tqdm import tqdm
    from skimage.measure import label, regionprops
    dataset = ImageFolderDataset('/home/wjj/work_space/fpn.segm/isaid_segm/train/images/',
                                '/home/wjj/work_space/fpn.segm/isaid_segm/train/masks/')
    cls_num_list = [0] * 16
    for img, mask, fpath in tqdm(dataset):
        for cls_idx in range(len(cls_num_list)):
            cls_num_list[cls_idx] += int(np.sum(mask == cls_idx))
        print(cls_num_list)
    total_num = float(sum(cls_num_list))
    print([cls_num / total_num for cls_num in cls_num_list])
    
    
    
    #connect_mask, obj_num= label(cls_mask, background=0, neighbors=4, connectivity=1, return_num=True)
    # props = regionprops(connect_mask)
    # for i in range(obj_num):
    #     print(props[i].area)
    #print(obj_num)
    #import matplotlib.pyplot as plt
    #plt.imshow(connect_mask)
    #plt.show()