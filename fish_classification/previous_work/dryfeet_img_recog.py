import torch
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
# from cnn import CNN, data_transforms
from torchvision import transforms
import torch.nn.functional as F
import tkinter
# torch.nn.Module.dump_patches = True

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
data_transforms = transforms.Compose([
    transforms.Scale(224),
    torchvision.transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


while True:
    # root = tkinter.Tk()
    # root.withdraw()
    name = tkinter.filedialog.askopenfilename()
    print(name)
    # root.update()
    im = Image.open(name)
    im_tensor = data_transforms(im)
    im_tensor.unsqueeze_(0)  # In-place version of unsqueeze()

    things = torch.load('./weights/88.save')
    # # things = torch.load('1000.save')
    cnn = things['model']
    cnn.cpu()
    #
    dset_classes = things['dset_classes']
    #
    # im = Image.open('./test_data/test.jpg')
    # im_tensor = data_transforms(im)
    # im_tensor.unsqueeze_(0)  # In-place version of unsqueeze()
    inputs = Variable(im_tensor)
    outputs = cnn(inputs)
    print(dset_classes)
    print(outputs)
    print(outputs.data)
    print(outputs.cpu().data.numpy())
    print(outputs.cpu().data.numpy()[0])
    print(outputs.cpu().data.topk(5,1,True,True))
    x = [1, 2, 3, 4, 5]
    y = outputs.cpu().data.numpy()[0]
    print(F.softmax(outputs.data.cpu().data.numpy()))
    y = F.softmax(outputs.data).cpu().data.numpy()[0]
    x_ticks_labels = dset_classes
    _, preds = torch.max(outputs.data, 1)
    # print(preds)
    # print('this is "'+dset_classes[preds]+'"')
    #
    # plt.ion()
    # out = torchvision.utils.make_grid(im_tensor)
    # # imshow(out, title=dset_classes[preds.cpu().numpy()[0][0]])
    # plt.ioff()
    # plt.show()


    plt.ion()
    out = torchvision.utils.make_grid(im_tensor)
    plt.rcdefaults()
    # plt.subplot.left=10
    # fig, ax = plt.subplots()
    fig = plt.figure(figsize=(5, 6))
    ax = plt.subplot(2,1,2)
    ax2 = plt.subplot(2,1,1)
    # fig.subplot.left = 10
    plt.gcf().subplots_adjust(left=0.3)
    # ax.plot(x,y)
    # ax.set_xticks(x)
    # # Set ticks labels for x-axis
    # ax.set_xticklabels(x_ticks_labels, rotation='vertical', fontsize=5)
    people = dset_classes
    y_pos = x
    performance = y
    # error = np.random.rand(len(people))

    ax.barh(y_pos, performance, align='center',
            color='green', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(people)
    ax.invert_yaxis()  # labels read top-to-bottom
    # ax.set_xlabel('Performance')
    # ax.set_title('How fast do you want to go today?')
    # imshow(out, title='ah')
    # out = out.numpy().transpose((1, 2, 0))
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # out = std * out + mean
    ax2.imshow(im)
    plt.ioff()
    plt.show()