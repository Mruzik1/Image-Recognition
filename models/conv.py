from torch import nn, Tensor


# typical convolution+pooling part of a CNN consisting of 3 blocks:
# "convolutional layer -> relu activation -> (max)pooling"
class ConvolutionalLayers(nn.Module):
    def __init__(self, hidden_units: int):
        super().__init__()

        self.conv1 = nn.Conv2d(3, hidden_units, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_units, hidden_units*2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_units*2, hidden_units*4, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()

    def forward(self, X: Tensor) -> Tensor:
        X = self.relu(self.conv1(X))
        X = self.pool(X)
        X = self.relu(self.conv2(X))
        X = self.pool(X)
        X = self.relu(self.conv3(X))
        return self.pool(X)
