#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of model from "End-to-end video background subtraction with 3d
convolutional neural networks" by Sakkos et al.

Created on Mon Feb 24, 2018
@author: Juan Terven
"""
import torch
import torch.nn as nn


class BackSubModel3d_2(nn.Module):
    def __init__(self):
        super(BackSubModel3d, self).__init__()

        self.crp1 = nn.Sequential(
                nn.Conv3d(3, 64, (3, 3, 3), stride=1, padding=(1, 1, 1)),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=(1, 2, 2))
                )

        self.crp2 = nn.Sequential(
                nn.Conv3d(64, 128, (3, 3, 3), stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=(1, 2, 2))
                )
        
        self.crp3 = nn.Sequential(
                nn.Conv3d(128, 256, (16, 3, 3), stride=1, padding=(0, 1, 1)),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=(1, 2, 2))
                )

        self.crp4 = nn.Sequential(
                nn.Conv2d(256, 512, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(512, 512, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(512, 512, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
                )

        self.cr = nn.Sequential(
                nn.Conv2d(512, 512, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(512, 512, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(512, 512, 3, stride=1, padding=1),
                nn.ReLU(),
                )

#        self.us1 = nn.ConvTranspose2d(256, 16, kernel_size=4,
#                                      stride=2, padding=1)
        self.us2 = nn.ConvTranspose2d(256, 16, kernel_size=8,
                                      stride=8, padding=0)
        self.us3 = nn.ConvTranspose2d(512, 16, kernel_size=16,
                                      stride=16, padding=0)
        self.us4 = nn.ConvTranspose2d(512, 16, kernel_size=16,
                                      stride=16, padding=0)

        self.fc = nn.Conv2d(48, 2, kernel_size=1)
        self.softmax = nn.LogSoftmax(1)

    def forward(self, x):
        """ Forward pass"""
        x1 = x[:, :, 0:4, :, :]
        x2 = x[:, :, 2:6, :, :]
        x3 = x[:, :, 4:8, :, :]
        x4 = x[:, :, 6:10, :, :]

#        print('x1:', x1.shape)

        crp1_1 = self.crp1_1(x1)
#        print('crp1_1:', crp1_1.shape)
        crp1_2 = self.crp1_1(x2)
#        print('crp1_2:', crp1_2.shape)
        crp1_3 = self.crp1_1(x3)
#        print('crp1_3:', crp1_3.shape)
        crp1_4 = self.crp1_1(x4)
#        print('crp1_4:', crp1_4.shape)

        # concatenate crp1_1 and crp1_2 in crp_12 and pass it to crp2_1
        crp_12 = torch.cat((crp1_1, crp1_2), dim=2)
#        print('crp_12:', crp_12.shape)
        crp2_1 = self.crp2_1(crp_12)
#        print('crp2_1:', crp2_1.shape)

        # concatenate crp1_3 and crp1_4 in crp_34 and pass it to crp2_2
        crp_34 = torch.cat((crp1_3, crp1_4), dim=2)
#        print('crp_34:', crp_34.shape)
        crp2_2 = self.crp2_2(crp_34)
#        print('crp2_2:', crp2_2.shape)

        # concatenate crp2_1 and crp2_2 in crp2 and input it to crp3
        crp2 = torch.cat((crp2_1, crp2_2), dim=2)
#        print('crp2:', crp2.shape)
        crp3 = self.crp3(crp2)
        #crp3 = crp3.view(1, 256, 30, 40)
        crp3 = torch.squeeze(crp3, dim=2)
#        print('crp3:', crp3.shape)

        crp4 = self.crp4(crp3)
#        print('crp4:', crp4.shape)
        cr = self.cr(crp4)
#        print('cr:', cr.shape)

        # Upsamplings
#        us1 = self.us1(crp2)
#        print('us1:', us1.shape)
        us2 = self.us2(crp3)
#        print('us2:', us2.shape)
        us3 = self.us3(crp4)
#        print('us3:', us3.shape)
        us4 = self.us4(cr)
#        print('us4:', us4.shape)

        # concatenate us1, us2, us3, us4 into us
        us = torch.cat((us2, us3, us4), dim=1)
#        print('us:', us.shape)
        out = self.fc(us)

#        print('fc:', out.shape)
        out = self.softmax(out)
#        print('out:', out.shape)

        return out
