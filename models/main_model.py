import torch
from torch.utils.data import DataLoader
from torch import nn, Tensor

from .classifier import Classifier
from .conv import ConvolutionalLayers


class MainModel:
    def __init__(self, classes_count: int, learning_rate: float = 0.1, size: tuple[int] = (64, 64)):
        hidden_units = 16                                                      # conv layers' hidden units
        classifier_input = hidden_units*size[0]//4*size[1]//4                  # classifier input size (depends on convolutional layer parameters, pooling, etc)

        self.__device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.__model = nn.Sequential(
            ConvolutionalLayers(hidden_units),
            Classifier(classifier_input, classes_count)
        ).to(self.__device)

        self.__optimizer = torch.optim.SGD(self.__model.parameters(), lr=learning_rate)
        self.__loss_fn = nn.CrossEntropyLoss()

        self.__train_history = []                                               # a list of loss values (appends with every new epoch)

    # getting training history
    @property
    def history(self) -> list[Tensor]:
        return self.__train_history
    
    # print training process
    def __print_process(self, loss: Tensor, batch: int, epoch: int, epochs_total: int):
        if (batch == 0):
            print('')
        print(f'Epoch: [{epoch+1}/{epochs_total}] | Batch: {batch} | Batch Loss: [{loss:.4f}]', end='\r') 

    # testing loop
    def test_loop(self, dataset: DataLoader) -> tuple[Tensor]:
        with torch.inference_mode():
            self.__model.eval()
            features, labels = next(iter(dataset))

            features.to(self.__device)
            labels.to(self.__device)

            pred = self.__model(features)
            loss = self.__loss_fn(pred, labels)

        self.__model.train()
        return pred.detach(), loss.detach()

    # training step (one epoch)
    def __train_step(self, epoch: int, epochs: int, dataset: DataLoader, total_loss: int) -> Tensor:
        for batch, (features, labels) in enumerate(dataset):
            features.to(self.__device)
            labels.to(self.__device)
            
            pred = self.__model(features)
            loss = self.__loss_fn(pred, labels)
            total_loss += loss.detach()
            
            self.__optimizer.zero_grad()
            loss.backward()
            self.__optimizer.step()

            self.__print_process(total_loss/len(dataset), batch, epoch, epochs)
        return total_loss/len(dataset)
    
    # training loop
    def train_loop(self, epochs: int, dataset: DataLoader):
        self.__model.train()
        print('Started Training...')

        for epoch in range(epochs):
            total_loss = self.__train_step(epoch, epochs, dataset, 0)
            
            print(f'\nTotal Epoch Loss: {total_loss:.4f}')
            self.__train_history.append(total_loss)