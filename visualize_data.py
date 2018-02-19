#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 06:24:15 2018

@author: juan
"""
import os
import cv2


def main():
    """ Visualize data from CDNet 2014 """
    data_path = '/datasets/backsub/cdnet2014/dataset'
    category_to_show = 'baseline'
    video_name = 'PETS2006'

    video_path = os.path.join(data_path, category_to_show, video_name, 'input')
    images_list = os.listdir(video_path)
    images_list.sort()

    cv2.namedWindow('Video')
    for frame in images_list:
        img = cv2.imread(os.path.join(video_path, frame))
        cv2.imshow('Video', img)
        cv2.waitKey(10)


def get_directories(path):
    """Return a list of directories name on the specifed path"""
    return [file for file in os.listdir(path)
            if os.path.isdir(os.path.join(path, file))]


if __name__ == "__main__":
    main()
