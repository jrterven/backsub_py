#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 18:04:56 2018

@author: juan
"""


def count_labels_distribution(data_loader):
    """ Count dataset statistics """
    bg_pix_count = 0
    fg_pix_count = 0
    for i, sample_batch in enumerate(data_loader):
        labels = sample_batch['gt'][:, :, :, :].numpy()
        bg_pix_count += (labels == 0).sum()
        fg_pix_count += (labels == 1).sum()

    return bg_pix_count, fg_pix_count
