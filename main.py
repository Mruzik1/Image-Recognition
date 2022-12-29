from data_preprocessor import DataHandler
from models import MainModel

import matplotlib.pyplot as plt
import torch, time

INIT_TIME = time.time()


if __name__ == '__main__':
    # a seed to decrease randomness
    # torch.manual_seed(100)


    # taking some data
    img_handler = DataHandler('./data')
    class_names = img_handler.class_names
    train_data, test_data = img_handler.get_data(batch_size=20, train_size=0.1)

    # creating/loading a model
    model_path = './model_data/cnn_model.pth'
    model = MainModel(len(class_names), learning_rate=0.05)

    # training a model
    model.train_loop(100, train_data, test_data, save_path=model_path)

    print(f'Training Time: {time.time() - INIT_TIME}')


    # visualize
    train_history, test_history = model.history
    
    ax = plt.subplot()

    ax.plot(train_history, label='training loss')
    ax.plot(test_history, label='testing loss')
    ax.set_xlabel('iterations number')
    ax.set_ylabel('loss per epoch')
    ax.legend()

    plt.show()