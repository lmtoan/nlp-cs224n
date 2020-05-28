#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class Highway(nn.Module):
    def __init__(self, in_dim):
        super(Highway, self).__init__()
        self.projection = nn.Linear(in_dim, in_dim)
        self.gate = nn.Linear(in_dim, in_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, X_conv):
        X_proj = F.relu(self.projection(X_conv))
        X_gate = torch.sigmoid(self.gate(X_conv))
        X_highway = X_gate * X_proj + (1 - X_gate) * X_conv
        X_highway = self.dropout(X_highway)
        return X_highway
