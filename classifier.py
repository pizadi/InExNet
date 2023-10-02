import torch
from torch import nn
from torch.nn import functional as F
from inexnet import InExNet

class Classifier(nn.Module):
    def __init__(
        self, 
        n_classes=10, 
        input_size=[128, 128],
        in_dims=3, 
        dims=[32, 64, 128, 256, 512, 512], 
        lin_dims=[1024, 256, 128]
        ):
        super().__init__()
        self.inexnet = InExNet(dims=dims, in_dims=in_dims)

        self.flatten = nn.Flatten()


        features = dims[-1] * input_size[0] * input_size[1] // (2**(2 * (len(dims)-1)))
        ldims = [features] + lin_dims
        self.linear = nn.ModuleList([nn.Sequential(
            nn.Linear(ldims[i], ldims[i+1]),
            nn.BatchNorm1d(ldims[i+1]),
            nn.GELU()
        ) for i in range(len(ldims) - 1)])
        self.out = nn.Linear(dims[-1], n_classes)
        self.outbn = nn.BatchNorm1d(n_classes)
    
    def forward(self, x):
        x = self.inexnet(x)[-1]
        x = self.flatten(x)
        for module in self.linear:
            x = module(x)
        x = self.outbn(self.out(x))

        return x

def inexnet_small(input_size=[128, 128], n_classes=10, in_dims=3):
    model = Classifier(
        input_size=input_size,
        n_classes=n_classes,
        in_dims=in_dims,
        dims=[32, 128, 256, 480],
        lin_dims=[256, 128, 64]
    )
    return model

def inexnet_medium(input_size=[128, 128], n_classes=10, in_dims=3):
    model = Classifier(
        input_size=input_size,
        n_classes=n_classes,
        in_dims=in_dims,
        dims=[64, 128, 256, 512, 512],
        lin_dims=[1024, 256, 128]
    )
    return model

def inexnet_large(input_size=[128, 128], n_classes=10, in_dims=3):
    model = Classifier(
        input_size=input_size,
        n_classes=n_classes,
        in_dims=in_dims,
        dims=[80, 320, 512, 640, 640],
        lin_dims=[1024, 512, 256, 256]
    )
    return model