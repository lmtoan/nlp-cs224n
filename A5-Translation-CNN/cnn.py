#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, in_dim, num_filters, kernel_size):
        super(CNN, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_dim, num_filters, kernel_size)

    def forward(self, X):
        X_conv = F.relu(self.conv(X))
        # pool = torch.max(X_conv, -1)
        # X_conv = pool.values
        X_conv = F.max_pool1d(X_conv, kernel_size=X_conv.shape[-1]).squeeze(dim=-1)
        return X_conv
