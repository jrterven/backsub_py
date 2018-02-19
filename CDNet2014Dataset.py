#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pytorch Dataloader for CDNet2014 dataset
http://changedetection.net/

Created on Sun Feb  4 07:01:42 2018

@author: Juan Terven
"""
from __future__ import print_function, division
import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset


class CDNet2014Dataset(Dataset):
    """CDNet 2014 dataset."""

    def __init__(self, root_dir, category, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            category (string): Category to use. E.g. 'baseline'
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.category = category
        self.transform = transform

        # build a list of images path and groundtruth paths with all the files
        # in the category
        category_path = os.path.join(root_dir, category)
        video_names = [file for file in os.listdir(category_path)
                       if os.path.isdir(os.path.join(category_path, file))]
        self.images_list = []
        self.gt_list = []
        for vid in video_names:
            # get the temporal ROI
            f = open(os.path.join(root_dir, category, vid, 'temporalROI.txt'))
            roi_str = f.read()
            f.close()
            t_roi = [int(n) for n in roi_str.split()]

            # exted the lists with the elements from the current video
            vid_path = os.path.join(root_dir, category, vid, 'input')
            gt_path = os.path.join(root_dir, category, vid, 'groundtruth')
            img_list = os.listdir(vid_path)
            img_list.sort()
            gt_list = os.listdir(gt_path)
            gt_list.sort()
            self.images_list.extend([os.path.join(vid_path, x)
                                     for x in img_list[t_roi[0]-1:t_roi[1]]])
            self.gt_list.extend([os.path.join(gt_path, x)
                                 for x in gt_list[t_roi[0]-1:t_roi[1]]])

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        img_name = self.images_list[idx]
        gt_name = self.gt_list[idx]

        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(gt_name, cv2.IMREAD_GRAYSCALE)
        image[gt == 85] = [0, 0, 0]
        gt2 = gt.copy()
        gt2[gt == 85] = 0
        gt2[gt != 255] = 0

        sample = {'image': image, 'gt': gt2}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h))
        gt = cv2.resize(gt, (new_w, new_h))

        return {'image': img, 'gt': gt}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]
        gt = gt[top: top + new_h, left: left + new_w]

        return {'image': image, 'gt': gt}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        gt = np.expand_dims(gt, axis=2)
        gt = gt.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'gt': torch.from_numpy(gt)}
