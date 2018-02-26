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


class BackSubModel13d(nn.Module):
    def __init__(self):
        super(BackSubModel13d, self).__init__()

        self.crp1_1 = nn.Sequential(
                nn.conv3d(3, 64, (3, 3, 4), stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=(2, 2, 1))
                )

        self.crp1_2 = nn.Sequential(
                nn.conv3d(3, 64, (3, 3, 4), stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=(2, 2, 1))
                )
        self.crp1_3 = nn.Sequential(
                nn.conv3d(3, 64, (3, 3, 4), stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=(2, 2, 1))
                )
        self.crp1_4 = nn.Sequential(
                nn.conv3d(3, 64, (3, 3, 4), stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=(2, 2, 1))
                )

        self.crp2_1 = nn.Sequential(
                nn.conv3d(64, 128, (3, 3, 2), stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=(2, 2, 1))
                )
        self.crp2_2 = nn.Sequential(
                nn.conv3d(64, 128, (3, 3, 2), stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=(2, 2, 1))
                )

        self.crp3 = nn.Sequential(
                nn.conv3d(128, 256, (3, 3, 16), stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=(2, 2, 1))
                )

        self.crp4 = nn.Sequential(
                nn.conv2d(256, 512, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.conv2d(256, 512, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.conv2d(256, 512, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
                )

        self.cr = nn.Sequential(
                nn.conv2d(512, 512, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.conv2d(512, 512, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.conv2d(512, 512, 3, stride=1, padding=1),
                nn.ReLU(),
                )

        self.fc = nn.Conv2d(512, 1, kernel_size=1)

        self.us1 = nn.ConvTranspose2d(256, 16, kernel_size=4,
                                      stride=2, padding=1)
        self.us2 = nn.ConvTranspose2d(256, 16, kernel_size=8,
                                      stride=4, padding=1)
        self.us3 = nn.ConvTranspose2d(512, 16, kernel_size=16,
                                      stride=8, padding=1)
        self.us4 = nn.ConvTranspose2d(512, 16, kernel_size=32,
                                      stride=16, padding=1)

    def forward(self, x):
        """ Forward pass"""
        crp1_1 = self.crp1_1(x[0])
        crp1_2 = self.crp1_1(x[1])
        crp1_3 = self.crp1_1(x[2])
        crp1_4 = self.crp1_1(x[3])

        # concatenate crp1_1 and crp1_2 in crp_12 and pass it to crp2_1
        crp_12 = torch.cat((crp1_1, crp1_2))
        crp2_1 = self.crp2_1(crp_12)

        # concatenate crp1_3 and crp1_4 in crp_34 and pass it to crp2_2
        crp_34 = torch.cat((crp1_3, crp1_4))
        crp2_2 = self.crp2_2(crp_34)

        # concatenate crp2_1 and crp2_2 in crp2 and input it to crp3
        crp2 = torch.cat((crp2_1, crp2_2))
        crp3 = self.crp3(crp2)
        crp4 = self.crp4(crp3)
        cr = self.cr(crp4)

        # Upsamplings
        us1 = self.us1(crp2)
        us2 = self.us2(crp3)
        us3 = self.us2(crp4)
        us4 = self.us4(cr)

        # concatenate us1, us2, us3, us4 into us
        us = torch.cat((us1, us2, us3, us4))
        out = self.fc(us)

        return out
