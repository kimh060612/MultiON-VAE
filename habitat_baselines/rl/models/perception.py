import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResidualBlock, self).__init__()
        
        self.conv_1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1)
        self.act_1 = nn.ReLU()
        self.skip_conv_1 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1)
        self.act_2 = nn.ReLU()
        self.skip_conv_2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1)
        self.act_3 = nn.ReLU()
    
    def forward(self, x):
        x = self.conv_1(x)
        x = self.act_1(x)
        _skip = x
        x = self.skip_conv_1(x)
        x = self.act_2(x)
        x = self.skip_conv_2(x)
        return self.act_3(x + _skip)

class ResidualDownSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResidualDownSample, self).__init__()
        
        self.conv_1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=2)
        self.skip_conv_1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=2, padding=1)
        self.act_1 = nn.ReLU()
        self.skip_conv_2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1)
        self.act_2 = nn.ReLU()
        
    def forward(self, x):
        _skip = x
        _skip = self.skip_conv_1(_skip)
        _skip = self.act_1(_skip)
        _skip = self.skip_conv_2(_skip)
        x = self.conv_1(x)
        return self.act_2(x + _skip)

class PerceptionCNN(nn.Module):
    def __init__(self, in_channel=3, hidden_channel = [16, 32], out_channel=9):
        super(PerceptionCNN, self).__init__()
        self.layers = nn.Sequential(
            ResidualBlock(in_channel, hidden_channel[0]),
            ResidualBlock(hidden_channel[0], hidden_channel[0]),
            ResidualDownSample(hidden_channel[0], hidden_channel[1]),
            ResidualBlock(hidden_channel[1], hidden_channel[1]),
            ResidualDownSample(hidden_channel[1], out_channel)
        )
        self.out_dim = out_channel
    
    def forward(self, x):
        x = self.layers(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), self.out_dim)
        return x