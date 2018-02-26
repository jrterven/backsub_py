#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 06:32:46 2018

@author: juan
"""

import cv2
import numpy as np

image = cv2.imread('/datasets/backsub/cdnet2014/dataset/baseline/highway/input/in000303.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
bg = cv2.imread('/datasets/backsub/cdnet2014/dataset/baseline/highway/background.jpg')
bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)

# calculate foreground
fg = cv2.absdiff(image,bg) # Absolute differences between the 2 images 
# set threshold to ignore small differences you can also use inrange function
_, fg = cv2.threshold(fg, 50, 255, 0)
img_c = cv2.merge([fg, bg, image])

cv2.imshow('img', image)
cv2.imshow('bg', bg)
cv2.imshow('fg', fg)
cv2.imshow('merge', img_c)

cv2.waitKey(0)