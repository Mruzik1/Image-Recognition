from torchvision import transforms, datasets


transform_fn = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

data = datasets.ImageFolder('./data', transform=transform_fn)

print(data[0][0].size())