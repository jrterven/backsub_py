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


class CDNet2014Dataset3d(Dataset):
    """CDNet 2014 dataset."""

    def __init__(self, root_dir, category, train, num_consecutive_frames=1,
                 transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            category (string): Category to use. E.g. 'baseline'
            train (boolean): train or test dataset
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.category = category
        self.num_frames = num_consecutive_frames
        self.transform = transform

        # build a list of images path and groundtruth paths with all the files
        # in the category
        category_path = os.path.join(root_dir, category)
        video_names = [file for file in os.listdir(category_path)
                       if os.path.isdir(os.path.join(category_path, file))]
        self.images_list = []
        self.gt_list = []

#        #for vid in video_names:
        vid = video_names[0]

        # get the temporal ROI
        f = open(os.path.join(root_dir, category, vid, 'temporalROI.txt'))
        roi_str = f.read()
        f.close()
        t_roi = [int(n) for n in roi_str.split()]

        test_size = (t_roi[1] - t_roi[0])//10
        if train:
            t_roi[1] = t_roi[1] - test_size
        else:
            t_roi[0] = t_roi[1] - test_size

        # extend the lists with the elements from the current video
        vid_path = os.path.join(root_dir, category, vid, 'input')
        gt_path = os.path.join(root_dir, category, vid, 'groundtruth')
        img_list = os.listdir(vid_path)
        img_list.sort()
        gt_list = os.listdir(gt_path)
        gt_list.sort()
        self.images_list.extend([os.path.join(vid_path, x)
                                 for x in img_list[t_roi[0]:t_roi[1]]])
        self.gt_list.extend([os.path.join(gt_path, x)
                             for x in gt_list[t_roi[0]:t_roi[1]]])

        self.data_len = len(self.images_list)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        if idx + self.num_frames >= self.data_len:
            idx = self.data_len - self.num_frames

        images = []
        labels = []
        for frame_idx in range(self.num_frames):
            img_name = self.images_list[idx]
            gt_name = self.gt_list[idx]
            image = cv2.imread(img_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            gt = cv2.imread(gt_name, cv2.IMREAD_GRAYSCALE)
            image[gt == 85] = 0

            gt2 = gt.copy()
            gt2[gt == 85] = 0
            gt2[gt != 255] = 0
            gt2[gt == 255] = 1

            image = image.astype(np.float32) / 255
            images.append(image)
            labels.append(gt2)

        #images_np = np.array(images)
        #labels_np = np.array(labels)
        sample = {'images': images, 'labels': labels}

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
        images, labels = sample['images'], sample['labels']

        for img_idx in range(len(images)):
            h, w = images[img_idx].shape[:2]
            if isinstance(self.output_size, int):
                if h > w:
                    new_h, new_w = self.output_size * h / w, self.output_size
                else:
                    new_h, new_w = self.output_size, self.output_size * w / h
            else:
                new_h, new_w = self.output_size

            new_h, new_w = int(new_h), int(new_w)

            images[img_idx] = cv2.resize(images[img_idx], (new_w, new_h))
            labels[img_idx] = cv2.resize(labels[img_idx], (new_w, new_h))

        images_np = np.array(images)
        labels_np = np.array(labels)

        return {'images': images_np, 'labels': labels_np}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        images, labels = sample['images'], sample['labels']

        # reorder axis because
        # numpy images: D x H x W x C
        # torch images: C x D x H X W
        images = images.transpose((3, 0, 1, 2))

        # expand and reorder because
        # labels images: D x H x W
        # torch images: C x D x H x W
        labels = np.expand_dims(labels, axis=3)
        labels = labels.transpose((3, 0, 1, 2))
        return {'images': torch.from_numpy(images),
                'labels': torch.from_numpy(labels)}
