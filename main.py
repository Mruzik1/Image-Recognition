from data_preprocessor import DataHandler
from models import MainModel

import matplotlib.pyplot as plt
import torch


if __name__ == '__main__':
    # a seed to decrease randomness
    # torch.manual_seed(100)

    # taking some data
    img_handler = DataHandler('./data')
    class_names = img_handler.class_names
    train_data, test_data = img_handler.get_data(batch_size=20, train_size=0.1)

    # creating a model
    model = MainModel(len(class_names), learning_rate=0.05)

    # training a model
    model.train_loop(15, train_data, test_data)

    train_history, test_history = model.history
    plt.plot(train_history)
    plt.plot(test_history)
    plt.show()

    # make some predictions
    pred, loss = model.test_loop(test_data)