from data_preprocessor import DataHandler


if __name__ == '__main__':
    img_handler = DataHandler('./data')
    class_names = img_handler.class_names
    train_data, test_data = img_handler.get_data()

    img, label = next(iter(test_data))
    print(img.size(), label)