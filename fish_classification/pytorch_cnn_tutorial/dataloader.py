from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

data_dir = './Batoidea(ga_oo_lee)'

transformation = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

# dataset = datasets.ImageFolder(data_dir)
# dataset = datasets.ImageFolder(data_dir, transform=ToTensor())
dataset = datasets.ImageFolder(data_dir, transformation)
num_classes = len(dataset.classes)
# loader = DataLoader(dataset)
# loader = DataLoader(dataset, shuffle=True)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

if __name__ == '__main__':
    # print(dataset.classes)
    # print(dataset.class_to_idx)
    # print(len(dataset))
    # print(dataset[0])
    # print(dataset[0][0])
    # print(dataset[0][1])
    # for i in range(20):
    #     x, y = dataset[i]
    #     print(y)
    #
    # print(type(dataset[0][0]))
    # print(type(dataset[0][1]))


    for x, y in loader:
        print(y)
        # print(x.shape)
        break

    print(loader)