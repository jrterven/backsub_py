#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 08:12:49 2018

@author: juan
"""
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
from CDNet2014Dataset3d import CDNet2014Dataset3d, Rescale, ToTensor
from model3d import BackSubModel3d


def main():
    dataset = CDNet2014Dataset3d(root_dir='/datasets/backsub/cdnet2014/dataset',
                               category='cameraJitter',
                               train=False,
                               num_consecutive_frames=10,
                               transform=transforms.Compose([
                                       Rescale((240, 320)),
                                       ToTensor()
                                       ]))
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=4)

     # Instantiate model
    model = BackSubModel3d()

    if torch.cuda.is_available():
        print('Using GPU:', torch.cuda.get_device_name(0))

        model.cuda()
    else:
        print('NO GPU DETECTED!')

    chk = '/home2/backsub_repo/checkpoints/model3d/model3d_camerajitter.pkl'
    print('Loading checkpoint ...')
    model.load_state_dict(torch.load(chk))

    for i_batch, sample_batch in enumerate(dataloader):
        print(i_batch, sample_batch['images'].size(),
              sample_batch['label'].size())

        images = sample_batch['images']

        if torch.cuda.is_available():
            images = Variable(images.cuda())
        else:
            images = Variable(images)

        # Forward pass only to get logits/output
        outputs = model(images)

        # Get predictions from the maximum value
        _, prediction = torch.max(outputs.data, 1)

        prediction = prediction.cpu().numpy()
        prediction = np.squeeze(prediction)
        prediction = prediction.astype(np.float32)
        print('output size:', prediction.shape)
        print(np.unique(prediction))

        frame = sample_batch['images'][0, :, 9, :, :].numpy().transpose((1, 2, 0))
        cv2.imshow('Video', frame)
        cv2.imshow('Pred', prediction)
    
        k = cv2.waitKey(10) & 0xFF
        if k == 27:
            break


if __name__ == "__main__":
    main()
