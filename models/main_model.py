import torch
from torch.utils.data import DataLoader
from torch import nn

from .classifier import Classifier
from .conv import ConvolutionalLayers


class MainModel:
    def __init__(self, classes_count: int, learning_rate: float = 0.1, img_size: tuple[int] = (64, 64)):
        self.__model = nn.Sequential(
            Classifier(classes_count),
            ConvolutionalLayers(img_size[0]*img_size[1])
        )
        self.__optimizer = torch.optim.SGD(self.__model.parameters(), learning_rate)
        self.__loss_fn = nn.CrossEntropyLoss()

        self.__train_history = []       # a list of loss values from every evaluation

    # getting training history
    @property
    def history(self) -> list[torch.Tensor]:
        return self.__train_history
    
    # print training process
    def __print_process(self, loss: float, batch: int, epoch: int, epochs_total: int):
        print(f'Epoch: [{epoch}/{epochs_total}] | Batch: {batch} | Total Loss: [{loss:.4f}]', end='\r')

        if (batch % (epochs_total // 5) == 0):
            print('', end='\n'*3 if epoch == epochs_total-1 else '\n')

    # testing loop
    def test_loop(self, dataset: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.inference_mode():
            self.__model.eval()
            features, labels = next(iter(dataset))

            pred = self.__model(features)
            loss = self.__loss_fn(pred, labels)

            self.__model.train()
        return pred.detach(), loss.detach()
    
    # training loop
    def train_loop(self, epochs: int, dataset: DataLoader):
        self.__model.train()

        for epoch in range(epochs):
            for batch, (features, labels) in enumerate(dataset):
                pred = self.__model(features)
                loss = self.__loss_fn(pred, labels)
                self.__train_history.append(loss.detach())

                loss.backward()
                self.__optimizer.zero_grad()
                self.__optimizer.step()

                self.__print_process(loss.detach(), batch, epoch, epochs)
                self.__train_history.append(loss.detach())
