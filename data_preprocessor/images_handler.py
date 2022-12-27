from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset, random_split

from os import cpu_count


class DataHandler:
    def __init__(self,  root: str):
        self.__transform_fn = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(size=(64, 64)),
            transforms.ToTensor()
        ])
        self.__img_folder = datasets.ImageFolder(root=root,
                                                 transform=self.__transform_fn)
    
    # getting class names
    @property
    def class_names(self) -> list[str]:
        return self.__img_folder.classes

    # getting class indexes
    @property
    def class_indexes(self) -> list[int]:
        return self.__img_folder.class_to_idx

    # returns training and testing subsets respectively
    def __split_data(self, train_part: float = 0.2) -> list[Subset]:
        test_n = int(len(self.__img_folder) * train_part)
        train_n = len(self.__img_folder)-test_n

        return random_split(self.__img_folder, (train_n, test_n))

    # returns training and testing data loaders respectively
    def get_data(self, batch_size: int = 1) -> tuple[DataLoader]:
        train, test = self.__split_data()

        train_dataloader = DataLoader(dataset=train, batch_size=batch_size,
                                      num_workers=cpu_count(), shuffle=True)

        test_dataloader = DataLoader(dataset=test, num_workers=cpu_count(), shuffle=True)

        return train_dataloader, test_dataloader