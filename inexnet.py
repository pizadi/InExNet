import torch
from torch import nn
from torch.nn import functional as F

class ResConv(nn.Module):
    def __init__(
        self,
        features,
        width
        ):
        """
        A single convolutional layer, with decoupled depthwise convolutions and a residual connection.
        Parameters:
        - features : The number of features in the conv layers
        - width : The kernel size of the pointwise convolution layer
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(features, features, width, padding='same', groups=features),
            nn.BatchNorm2d(features),
            nn.GELU(),
            nn.Conv2d(features, features, 1),
            nn.BatchNorm2d(features),
            nn.GELU()
        )

    def forward(self, x):
        return x + self.net(x)

class ConvBlock(nn.Module):
    def __init__(
        self,
        depth,
        features,
        width
        ):
        """
        A chain of convolutional layers.
        Parameters:
        - depth : The number of Convolutional layers in the ConvBlock
        - features : The number of features in the conv layers
        - width : The kernel size of the pointwise convolution layer
        """
        super().__init__()
        self.net = nn.Sequential(*[ResConv(features, width) for _ in range(depth)])

    def forward(self, x):
        x = self.net(x)
        return x

class IEBlock(nn.Module):
    def __init__(
        self,
        features
        ):
        """
        An inhibition-excitation block. Bilaterally inhibits and excites the features
        in two feature groups.
        Parameters:
        - features : The number of features in each of the groups
        """
        super().__init__()
        self.ie1 = nn.Conv2d(features, features, 1)
        self.ie2 = nn.Conv2d(features, features, 1)
        self.bn1 = nn.BatchNorm2d(features)
        self.bn2 = nn.BatchNorm2d(features)

    def forward(self, x, y):
        iex = torch.tanh(self.bn1(self.ie1(y)))
        iey = torch.tanh(self.bn2(self.ie2(x)))
        x = x + x * iex
        y = y + y * iey
        return x, y

class ProcBlock(nn.Module):
    def __init__(
        self,
        in_features,
        features
        ):
        """
        An feature-processing block which processes the features in two parallel
        wide and deep paths.
        Parameters:
        - in_features : The number of input features
        - features : The number of features in the output
        """
        super().__init__()
        self.initconv1 = nn.Conv2d(in_features, features // 2, 1)
        self.initconv2 = nn.Conv2d(in_features, features // 2, 1)
        self.proc1 = ConvBlock(5, features // 2, 3)
        self.proc2 = ConvBlock(3, features // 2, 7)
        self.mix = IEBlock(features // 2)
    
    def forward(self, x):
        x1 = self.initconv1(x)
        x2 = self.initconv2(x)
        x1 = self.proc1(x1)
        x2 = self.proc2(x2)
        x1, x2 = self.mix(x1, x2)
        x = torch.cat([x1, x2], dim=1)
        return x

class InExNet(nn.Module):
    """
    The InExNet backbone model.
    Parameters:
    - dims = The number of features in each block
    - input_dim = The number of channels in the input
    """
    def __init__(
        self,
        dims=[32, 64, 128, 256, 512, 512],
        input_dim=3
        ):
        super().__init__()
        self.initconv = nn.Conv2d(input_dim, dims[0], 3, padding='same')
        self.bn0 = nn.BatchNorm2d(dims[0])

        self.blocks = nn.ModuleList([nn.Sequential(
            ProcBlock(dims[i], dims[i+1]),
            nn.AvgPool2d(2, 2)
        ) for i in range(len(dims)-1)])
        

    def forward(self, x):
        outs = []
        x = F.gelu(self.bn0(self.initconv(x)))
        outs.append(x)

        for block in self.blocks:
            x = block(x)
            outs.append(x)

        return outs