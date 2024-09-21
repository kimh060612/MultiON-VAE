import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from habitat_baselines.common.utils import Flatten

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
    def __init__(self, observation_space, in_channel = 4, hidden_channel = [32, 64], out_channel = 32):
        super(PerceptionCNN, self).__init__()
        if "rgb" in observation_space.spaces:
            self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
        else:
            self._n_input_rgb = 0

        if "depth" in observation_space.spaces:
            self._n_input_depth = observation_space.spaces["depth"].shape[2]
        else:
            self._n_input_depth = 0
        self.layers = nn.Sequential(
            # ResidualBlock(in_channel, hidden_channel[0]),
            ResidualDownSample(in_channel, hidden_channel[0]),
            # ResidualBlock(hidden_channel[1], hidden_channel[1]),
            ResidualDownSample(hidden_channel[0], hidden_channel[1]),
            # ResidualBlock(hidden_channel[2], hidden_channel[3]),
            ResidualDownSample(hidden_channel[1], out_channel),
        )
        self.out_dim = out_channel
    
    @property
    def is_blind(self):
        return self._n_input_rgb + self._n_input_depth == 0
    
    def forward(self, observations):
        cnn_input = []
        if self._n_input_rgb > 0:
            rgb_observations = observations["rgb"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            rgb_observations = rgb_observations / 255.0  # normalize RGB
            cnn_input.append(rgb_observations)

        if self._n_input_depth > 0:
            depth_observations = observations["depth"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)
            cnn_input.append(depth_observations)

        x = torch.cat(cnn_input, dim=1)
        x = self.layers(x) # output_dim: N X 32 X 32 X 32
        return x
    
class TrajMapCNN(nn.Module):
    r"""A Simple 3-Conv CNN followed by a fully connected layer

    Takes in observations and produces an embedding of the occupancy map

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
    """

    def __init__(self, map_size, output_size, agent_type):
        super().__init__()
       
        self._n_input_map = 34


        if agent_type in ["oracle", "oracle-ego", "no-map"]:
            # kernel size for different CNN layers
            self._cnn_layers_kernel_size = [(4, 4), (3, 3), (2, 2)]
            # strides for different CNN layers
            self._cnn_layers_stride = [(2, 2), (1, 1), (1, 1)]
        else:
            self._cnn_layers_kernel_size = [(6, 6), (4, 4), (2, 2)]
            self._cnn_layers_stride = [(3, 3), (2, 2), (1, 1)]

        cnn_dims = np.array(
            [map_size, map_size], dtype=np.float32
        )
         
    
        if self.is_blind:
            self.cnn = nn.Sequential()
        else:
            for kernel_size, stride in zip(
                self._cnn_layers_kernel_size, self._cnn_layers_stride
            ):
                cnn_dims = self._conv_output_dim(
                    dimension=cnn_dims,
                    padding=np.array([0, 0], dtype=np.float32),
                    dilation=np.array([1, 1], dtype=np.float32),
                    kernel_size=np.array(kernel_size, dtype=np.float32),
                    stride=np.array(stride, dtype=np.float32),
                )

            self.cnn = nn.Sequential(
                nn.Conv2d(
                    in_channels=self._n_input_map,
                    out_channels=32,
                    kernel_size=self._cnn_layers_kernel_size[0],
                    stride=self._cnn_layers_stride[0],
                ),
                nn.ReLU(True),
                nn.Conv2d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=self._cnn_layers_kernel_size[1],
                    stride=self._cnn_layers_stride[1],
                ),
                nn.ReLU(True),
                nn.Conv2d(
                    in_channels=64,
                    out_channels=32,
                    kernel_size=self._cnn_layers_kernel_size[2],
                    stride=self._cnn_layers_stride[2],
                ),
                #  nn.ReLU(True),
                Flatten(),
                nn.Linear(32 * cnn_dims[0] * cnn_dims[1], output_size),
                nn.ReLU(True),
            )

        self.layer_init()

    def _conv_output_dim(
        self, dimension, padding, dilation, kernel_size, stride
    ):
        r"""Calculates the output height and width based on the input
        height and width to the convolution layer.

        ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
        """
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(
                    np.floor(
                        (
                            (
                                dimension[i]
                                + 2 * padding[i]
                                - dilation[i] * (kernel_size[i] - 1)
                                - 1
                            )
                            / stride[i]
                        )
                        + 1
                    )
                )
            )
        return tuple(out_dimension)

    def layer_init(self):
        for layer in self.cnn:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    @property
    def is_blind(self):
        return self._n_input_map == 0 ##alt
        

    def forward(self, observations):
        if self._n_input_map > 0:
            map_observations = observations.permute(0, 3, 1, 2)
        return self.cnn(map_observations)