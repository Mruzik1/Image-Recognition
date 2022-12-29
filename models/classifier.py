from torch import nn, Tensor


class Classifier(nn.Module):
    def __init__(self, input_size: int, classes_count: int):
        super().__init__()

        self.__model = nn.Sequential(
            nn.Flatten(),
            # nn.Linear(input_size, input_size//2),
            # nn.ReLU(),
            # nn.Linear(input_size//2, input_size//3),
            # nn.ReLU(),
            # nn.Linear(input_size//3, classes_count),
            # nn.Softmax(dim=1)

            nn.Linear(input_size, classes_count),
            nn.Softmax(dim=1)
        )

    def forward(self, X: Tensor) -> Tensor:
        return self.__model(X)