from torch import nn, Tensor


class Classifier(nn.Module):
    def __init__(self, input_size: int, classes_count: int):
        super().__init__()

        self.__model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, classes_count)
        )

    def forward(self, X: Tensor) -> Tensor:
        return self.__model(X)