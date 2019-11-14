from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),  # HxWxC -> CxHxW, [28x28] -> [1x28x28], [0,255] -> [0,1]
    transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
# Match directory path
train_dataset = datasets.MNIST(root='../../__SSSSShare_DDDate_set/mnist_data/', train=True,
                               transform=transform,
                               download=True)
test_dataset = datasets.MNIST(root='../../__SSSSShare_DDDate_set/mnist_data/', train=False,
                              transform=transform,
                              download=False)
