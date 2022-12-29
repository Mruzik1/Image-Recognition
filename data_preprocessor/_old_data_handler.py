from ._old_images_reader import ImagesReader

from torchvision import transforms
import torch


# I DON'T USE THIS CLASS ANYMORE (torchvision already has ImageFolder for my purposes)
# Also it's unfinished :)

class DataHandler:
    def __init__(self, size: tuple[int] = (64, 64)):
        self.__image_reader = ImagesReader()
        self.__images = self.__image_reader.read_images()
        self.__transform_fn = transforms.Compose([              # a function for transforming images
            transforms.Resize(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    # shuffles images and labels (same order correlation)
    def __shuffle_dataset(self, images: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor]:
        new_indexes = torch.randperm(images.size()[0])
        return images[new_indexes], labels[new_indexes]        
    
    # returns transformed images (rechanged, transformed to tensors, randomly flipped)
    # returns labels as numbers
    def get_labeled_images(self) -> tuple[torch.Tensor]:
        t_images = torch.Tensor()
        labels = torch.Tensor()
        
        for key, num in zip(self.__images, range(len(self.__images.keys()))):
            for img in self.__images[key]:
                t_img = self.__transform_fn(img)
                labels = torch.cat((labels, torch.tensor([num])))
                t_images = torch.cat((t_images, t_img.reshape((t_img.size()[:-1]))))

            print(t_images.size(), labels.size())
        
        return t_images, labels

    # getting name of the image classes
    @property
    def classnames(self) -> list[str]:
        return list(self.__images.keys())