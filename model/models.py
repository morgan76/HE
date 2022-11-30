# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F


class EmbedNet(nn.Module):
    def __init__(self, config):
        super(EmbedNet, self).__init__()
        self.use_batch_norm = config.use_batch_norm
        self.use_dropout = config.use_dropout
        conv_kernel_size = (6,4)
        self.pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.conv1 = nn.Conv2d(in_channels=1,
                            out_channels=64,
                            kernel_size=conv_kernel_size,
                            padding=(2, 1)
                            )
        self.conv2 = nn.Conv2d(in_channels=64,
                            out_channels=128,
                            kernel_size=conv_kernel_size,
                            padding=(2, 1)
                            )
        self.conv3 = nn.Conv2d(in_channels=128,
                            out_channels=256,
                            kernel_size=conv_kernel_size,
                            padding=(2, 1)
                            )
        

        if config.feat_id == 'cqt':
            self.fc1 = nn.Linear(12288, 128)
        elif config.feat_id == 'mel':
            self.fc1 = nn.Linear(10240, 128) 

        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)

        if self.use_dropout:
            self.dropout1 = nn.Dropout(0.1)
            self.dropout2 = nn.Dropout(0.1)

        if self.use_batch_norm:
            self.batch_norm1 = nn.BatchNorm2d(1)
            self.batch_norm2 = nn.BatchNorm2d(64)
            self.batch_norm3 = nn.BatchNorm2d(128)
            self.batch_norm4 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv1(x)
        if self.use_batch_norm:
            x = self.batch_norm2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 4))
        x = self.pad(x)
        x = self.conv2(x)
        if self.use_batch_norm:
            x = self.batch_norm3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (3, 4))
        x = self.pad(x)
        x = self.conv3(x)
        if self.use_batch_norm:
            x = self.batch_norm4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 4))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        if self.use_dropout:
            x = self.dropout2(x)
        x = self.fc2(x)
        x = F.relu(x)
        if self.use_dropout:
            x = self.dropout1(x)
        output = self.fc3(x)
        # L2 Normalization
        output_normalized = F.normalize(output, p=2, dim=-1)
        return output_normalized