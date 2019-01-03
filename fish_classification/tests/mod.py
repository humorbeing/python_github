from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(
    mean=mean,
    std=std
)
data_transforms = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ]
)


def im_to_tensor(image):
    return data_transforms(image)


def im_show(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    m = np.array(mean)
    s = np.array(std)
    inp = s * inp + m
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def model(name):
    load_from = './weights/'
    load_this = [
        'super_with_most_[0.98]_mean[0.95]_std[0.09].save',
        'shark_[0.99]_mean[0.98]_std[0.01].save',
        'Batoidea(ga_oo_lee)_[0.99]_mean[0.99]_std[0.01].save',
        'tuna_fish_[0.95]_mean[0.93]_std[0.05].save',
        # 'super_class_[1.0]_mean[0.99]_std[0.0].save',
        # 'all_in_one_[0.98]_mean[0.96]_std[0.07].save'
    ]
    name_to_num = {
        'all': 0,
        'shark': 1,
        'gaa': 2,
        'tuna': 3
    }
    loaded = torch.load(load_from + load_this[name_to_num[name]])
    mo = loaded['model']
    mo.cpu()
    classes = loaded['dset_classes']
    performance = loaded['performance']
    return mo, classes, performance


def get_im_var(name='test.jpg', load_from='./'):
    a = '/media/ray/SSD/workspace/python/fish_classification/storage/datasets/korea_fish/Oncorhynchus keta/2.jpg'
    im = Image.open(a)
    # im = Image.open(load_from + name)
    im_tensor = im_to_tensor(im)
    im_tensor.unsqueeze_(0)
    inputs = Variable(im_tensor)
    return inputs


def get_output(model, inputs):
    outputs = model(inputs)
    y = F.softmax(outputs).cpu().data.numpy()[0]
    return y


def get_order(y):
    s = np.array(y)
    s.sort()
    order = []
    a = np.where(y == s[-1])[0][0]
    order.append(a)
    a = np.where(y == s[-2])[0][0]
    order.append(a)
    a = np.where(y == s[-3])[0][0]
    order.append(a)
    return order


def get_string(classes, performance, y, s):
    num_to_class = {
        2: 'gaa',
        12: 'shark',
        13: 'tuna'
    }
    whole = []
    for i in range(3):
        line = []
        line.append(classes[s[i]]+' ')
        line.append(str(int(performance[s[i]] * 100))+' ')
        line.append(str(int(y[s[i]] * 10000)))
        whole.append(line)


    if s[0] in num_to_class:
        cnn, classes, performance = model(num_to_class[s[0]])
        inputs = get_im_var()
        y = get_output(cnn, inputs)
        print(y)
        s = get_order(y)
        print(s)
        # whole = []
        for i in range(3):
            line = []
            line.append(classes[s[i]]+' ')
            line.append(str(int(performance[s[i]] * 100))+' ')
            line.append(str(int(y[s[i]] * 10000)))
            whole.append(line)

    # print(whole)
    return whole