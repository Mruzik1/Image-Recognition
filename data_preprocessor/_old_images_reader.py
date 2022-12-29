import os
from typing import Iterator
from PIL import Image


# I DON'T USE THIS CLASS ANYMORE (torchvision already has ImageFolder for my purposes)

class ImagesReader:
    def __init__(self, root: str ='./data'):
        self.__root_path = root
        self.__inner_dirs = next(os.walk(root))[1]

    # getting names of the inner directories
    @property
    def inner_dirs(self) -> list[str]:
        return self.__inner_dirs

    # returns minimum number of images in a folder
    def __min_images(self) -> int:
        return min([len(os.listdir(f'{self.__root_path}/{dir}')) for dir in self.inner_dirs])

    # parses images inside a folder, transforms images to the numpy array
    def __parse_directory(self, path: str) -> Iterator:
        images = os.listdir(path)[:self.__min_images()]

        for idx, img in enumerate(images):
            print(f'Reading images from "{path}" [{idx}/{len(images)}]...'+' '*30, end='\r')
            yield Image.open(f'{path}/{img}')
        print(f'"{path}" is done!'+' '*30)

    # checks all inner folders, returns a dictionary
    def read_images(self) -> dict[str, Iterator]:
        return {k: self.__parse_directory(f'{self.__root_path}/{k}') for k in self.inner_dirs}