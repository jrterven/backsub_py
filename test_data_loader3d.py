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
                               train=False,
                               num_consecutive_frames=10,
                               transform=transforms.Compose([
                                       Rescale((240, 320)),
                                       ToTensor()
                                       ]))
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=True, num_workers=4)

    do_exit = False
    for i_batch, sample_batch in enumerate(dataloader):
        print(i_batch, sample_batch['images'].size(),
              sample_batch['label'].size())

        gt = sample_batch['label'][0, :, :, :].numpy().transpose((1, 2, 0))
        gt = gt * 255
        cv2.imshow('gt', gt)
            
        num_frames = sample_batch['images'].shape[2]
        print(num_frames)
        for i in range(num_frames):
            img = sample_batch['images'][0, :, i, :, :].numpy().transpose((1, 2, 0))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            cv2.imshow('image', img)
    
            k = cv2.waitKey(0) & 0xFF
            if k == 27:
                do_exit = True
                break
            
        if do_exit:
            break
        #cv2.waitKey(500)

if __name__ == "__main__":
    main()
