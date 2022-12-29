from torch import nn, Tensor


# fully-connected neural network 
# it's enough for my CNN
class Classifier(nn.Module):
    def __init__(self, input_size: int, classes_count: int):
        super().__init__()

        self.__model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, classes_count),
            nn.Softmax(dim=1)
        )

    def forward(self, X: Tensor) -> Tensor:
        return self.__model(X)