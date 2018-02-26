#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 10:34:53 2018

@author: juan
"""
import torch
import torch.nn as nn
from torch.autograd import Variable

N, C = 5, 4
loss = nn.NLLLoss2d()
# input is of size N x C x height x width
data = Variable(torch.randn(N, 16, 10, 10))
m = nn.Conv2d(16, C, (3, 3))
# each element in target has to have 0 <= value < C
target = Variable(torch.LongTensor(N, 8, 8).random_(0, C))
out = m(data)

print('out:', out.shape)
print('target:', target.shape)

output = loss(out, target)
output.backward()