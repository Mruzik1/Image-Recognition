from data_preprocessor import DataHandler
from models import MainModel

import matplotlib.pyplot as plt
import torch


if __name__ == '__main__':
    # a seed to decrease random
    torch.manual_seed(100)

    # taking some data
    img_handler = DataHandler('./data')
    class_names = img_handler.class_names
    train_data, test_data = img_handler.get_data(batch_size=20, train_size=0.2)

    # creating a model
    model = MainModel(len(class_names))

    # training a model
    model.train_loop(3, train_data)

    plt.plot(model.history)
    plt.show()

    # make some predictions
    pred = model.test_loop(test_data)[0][0]
    img = next(iter(test_data))[0].squeeze()

    # visualization
    plt.imshow(img.transpose(0, 2))
    plt.show()
    print('\n', class_names[pred.argmax()])


    # testing the model with a dummy data
    # dummy_img = torch.randn(size=(1, 3, 64, 64))
    # mm = MainModel(5)

    # with torch.inference_mode():
    #     print(mm.model(dummy_img).size())