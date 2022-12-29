import torch
from torch.utils.data import DataLoader
from torch import nn, Tensor

from .classifier import Classifier
from .conv import ConvolutionalLayers


class MainModel:
    def __init__(self, classes_count: int, learning_rate: float = 0.1,
                 size: tuple[int] = (64, 64), load_path: str = None):

        hidden_units = 16                                                      # conv layers' hidden units
        classifier_input = hidden_units*size[0]//8*size[1]//8                  # classifier input size (depends on the convolution+pooling parameters)

        self.__device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.__model = nn.Sequential(
            ConvolutionalLayers(hidden_units),
            Classifier(classifier_input, classes_count)
        ).to(self.__device)

        if (load_path != None):
            self.__model.load_state_dict(torch.load(f=load_path))

        self.__optimizer = torch.optim.SGD(self.__model.parameters(), lr=learning_rate)
        self.__loss_fn = nn.CrossEntropyLoss()

        self.__train_history = []                                               # lists of loss values (appends with every new epoch)
        self.__test_history = []                                                #
        self.__accuracy_history = []                                            # 

    # getting training, testing, and accuracy histories respectively
    @property
    def history(self) -> tuple[list[Tensor]]:
        return self.__train_history, self.__test_history, self.__accuracy_history
    
    # testing loop
    def test_loop(self, dataset: DataLoader) -> tuple[Tensor, Tensor]:
        total_loss = 0
        total_accuracy = 0

        self.__model.eval()
        with torch.inference_mode():
            for features, labels in dataset:
                features, labels = features.to(self.__device), labels.to(self.__device)

                pred = self.__model(features)
                loss = self.__loss_fn(pred, labels)

                total_accuracy += torch.sigmoid(pred).argmax() == labels
                total_loss += loss

        return (total_accuracy/len(dataset))[0], total_loss/len(dataset)

    # training step (one epoch)
    def __train_step(self, epoch: int, epochs: int, dataset: DataLoader, total_loss: int) -> Tensor:
        for batch, (features, labels) in enumerate(dataset):
            features, labels = features.to(self.__device), labels.to(self.__device)
            
            pred = self.__model(features)
            loss = self.__loss_fn(pred, labels).to(self.__device)
            total_loss += loss.detach()
            
            self.__optimizer.zero_grad()
            loss.backward()
            self.__optimizer.step()

            print(f'Epoch: [{epoch+1}/{epochs}] | Batch: [{batch}] | Batch Loss: [{total_loss/len(dataset):.4f}]', end='\r')

        return total_loss/len(dataset)
    
    # training loop
    def train_loop(self, epochs: int, dataset: DataLoader, test_dataset: DataLoader, save_path: str = None):
        print('Started Training...\n')

        for epoch in range(epochs):
            self.__model.train()
            total_loss = self.__train_step(epoch, epochs, dataset, 0)
            accuracy, test_loss = self.test_loop(test_dataset)
            
            print(f'\nTotal Epoch Loss: [{total_loss:.4f}]')
            self.__test_history.append(test_loss)

            print(f'Total Testing Loss: [{self.__test_history[-1]:.4f}] | Testing Accuracy: [{accuracy*100:.2f}%]\n')
            self.__train_history.append(total_loss)
            self.__accuracy_history.append(accuracy)
        
        if save_path != None:
            torch.save(obj=self.__model.state_dict(), f=save_path)