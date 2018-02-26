#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 08:12:49 2018

@author: juan
"""
import cv2
from torch.utils.data import DataLoader
from torchvision import transforms
from CDNet2014Dataset3d import CDNet2014Dataset3d, Rescale, ToTensor


def main():
    dataset = CDNet2014Dataset3d(root_dir='/datasets/backsub/cdnet2014/dataset',
                               category='baseline',
                               train=True,
                               num_consecutive_frames=10,
                               transform=transforms.Compose([
                                       Rescale((240, 320)),
                                       ToTensor()
                                       ]))
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=4)

    for i_batch, sample_batch in enumerate(dataloader):
        print(i_batch, sample_batch['images'].size(),
              sample_batch['labels'].size())

        img = sample_batch['images'][0, :, 0, :, :].numpy().transpose((1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        gt = sample_batch['labels'][0, :, 0, :, :].numpy().transpose((1, 2, 0))
        gt = gt * 255

        cv2.imshow('image', img)
        cv2.imshow('gt', gt)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break


if __name__ == "__main__":
    main()
