#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 14:13:54 2018

@author: juan
"""
import torch.nn as nn


class BackSubModel1(nn.Module):
    def __init__(self):
        super(BackSubModel1, self).__init__()

        num_filters = 64
        self.encoder = nn.Sequential(
                nn.Conv2d(3, num_filters, 3, padding=1),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(num_filters, num_filters*2, 3, padding=1),
                nn.BatchNorm2d(num_filters*2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(num_filters*2, num_filters*2, 3, padding=1),
                nn.BatchNorm2d(num_filters*2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(num_filters*2, num_filters*4, 3, padding=1),
                nn.BatchNorm2d(num_filters*4),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                )
        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(num_filters*4, num_filters*4, kernel_size=4,
                                   stride=2, padding=1),
                nn.BatchNorm2d(num_filters*4),
                nn.ReLU(),
                nn.ConvTranspose2d(num_filters*4, num_filters*4, kernel_size=4,
                                   stride=2, padding=1),
                nn.BatchNorm2d(num_filters*4),
                nn.ReLU(),
                nn.ConvTranspose2d(num_filters*4, num_filters*4, kernel_size=4,
                                   stride=2, padding=1),
                nn.BatchNorm2d(num_filters*4),
                nn.ReLU(),
                nn.ConvTranspose2d(num_filters*4, num_filters*4, kernel_size=4,
                                   stride=2, padding=1),
                nn.BatchNorm2d(num_filters*4),
                nn.ReLU(),
                )
        self.conv1x1 = nn.Conv2d(num_filters*4, 2, kernel_size=1)
        self.softmax = nn.LogSoftmax(1)

    def forward(self, x):
        """ Forward pass"""
        out = self.encoder(x)
        out = self.decoder(out)
        out = self.conv1x1(out)
        out = self.softmax(out)

        return out
