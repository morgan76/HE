import torch
import torch.nn as nn
import torch.nn.functional as F

class CS_Tripletnet(nn.Module):
    def __init__(self, embeddingnet):
        super(CS_Tripletnet, self).__init__()
        self.embeddingnet = embeddingnet



    def forward(self, x, y, z, c):
        """ x: Anchor image,
            y: Distant (negative) image,
            z: Close (positive) image,
            c: Integer indicating according to which notion of similarity images are compared"""
        embedded_x = self.embeddingnet(x, c)
        embedded_y = self.embeddingnet(y, c)
        embedded_z = self.embeddingnet(z, c)
        return embedded_x, embedded_y, embedded_z

    