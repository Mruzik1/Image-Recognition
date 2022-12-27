from torch import nn


class ConvolutionalLayers(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()

        self.conv1 = nn.Conv2d()
        self.conv2 = nn.Conv2d()

        self.relu = nn.ReLU()