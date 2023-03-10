from data_preprocessor import DataHandler
from models import MainModel

import matplotlib.pyplot as plt
import torch, time

INIT_TIME = time.time()


if __name__ == '__main__':
    # a seed to decrease randomness
    torch.manual_seed(100)


    # taking some data
    img_size = (80, 80)
    img_handler = DataHandler('./data', size=img_size)
    class_names = img_handler.class_names
    train_data, test_data = img_handler.get_data(batch_size=20, test_size=0.08)

    # creating/loading a model
    model_path = './model_data/cnn_model2.pth'
    model = MainModel(len(class_names), learning_rate=0.05, size=img_size, load_path=model_path)
    

    # training a model
    # model.train_loop(0, train_data, test_data)

    # print(f'Training Time: {time.time() - INIT_TIME}')


    # evaluating a model
    acc, loss = model.test_loop(test_data)
    print(f'Accuracy: [{acc*100:.2f}%] | Test Loss: [{loss:.4f}]')


    # visualize
    # train_history, test_history, accuracy = model.history
    # fig, axs = plt.subplots(2)
    # fig.set_size_inches(12, 9.5)

    # axs[0].set_title('Total Loss')
    # axs[0].plot(train_history, label='training loss')
    # axs[0].plot(test_history, label='testing loss')
    # axs[0].set_xlabel('Iterations Number')
    # axs[0].set_ylabel('Loss per Epoch')
    # axs[0].legend()

    # axs[1].set_title('Total Testing Accuracy')
    # axs[1].plot(accuracy)
    # axs[1].set_xlabel('Iterations Number')
    # axs[1].set_ylabel('Accuracy per Epoch')

    # plt.show()