import argparse
import os
import sys
import time
import re

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx
import torchvision
import utils
from transformer_net import TransformerNet
from vgg import Vgg16
from PIL import Image
device = torch.device('cpu')
device = torch.device('cuda:0')
def stylize_one(style_model_path, target_image):
    content_image = utils.load_image(target_image)
    # print('content_image', content_image)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)
    with torch.no_grad():

        style_model = TransformerNet()
        state_dict = torch.load(style_model_path)

        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]

        style_model.load_state_dict(state_dict)
        style_model.to(device)
        output = style_model(content_image)
        data = output[0].clamp(0, 255)
        # torchvision.utils.save_image(data, './1.png', normalize=True)
        img = data.cpu().clone().clamp(0, 255).numpy()
        img = img.transpose(1, 2, 0).astype("uint8")
        img = Image.fromarray(img)
    return img, data


def ta(model_name, image, model_root=''):
    model_path = model_root + model_name + '.model'
    img, data = stylize_one(model_path, image)
    return img, data

if __name__ == "__main__":
    # model_name = '01'

    # model_path = '/home/ray/Desktop/Link to Mystuff/Workspace/python_world/python_github/__SSSSTTTTOOOORRRREEEE/neural-style/saved-model-here/style_05.model'
    model_root = '/home/ray/Desktop/Link to Mystuff/Workspace/python_world/python_github/__SSSSTTTTOOOORRRREEEE/neural-style/saved-model-here/trained on colab/'
    # target_image_path = '/home/ray/Desktop/Link to Mystuff/Workspace/python_world/python_github/__SSSSTTTTOOOORRRREEEE/neural-style/1/WeChat Image_20200118161228.jpg'
    # target_image_path = '/home/ray/Desktop/Link to Mystuff/Workspace/python_world/python_github/__SSSSTTTTOOOORRRREEEE/neural-style/all content/content/20181007_131604.jpg'
    save_path = './img/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # img = ta(model_name, target_image_path, model_root)
    # img.show()
    content_root = '/home/ray/Desktop/Link to Mystuff/Workspace/python_world/python_github/__SSSSTTTTOOOORRRREEEE/neural-style/all family image/family/'
    from os import walk

    f = []
    for (dirpath, dirnames, filenames) in walk(content_root):
        f.extend(filenames)
        break
    # area = []

    for cc in f:
        target_image_path = content_root + cc
        c_name = cc[:-4]
        print(c_name)
        rows = []
        for i in range(1,11):
            num = '{:02d}'.format(i)
            img, data = ta(num, target_image_path, model_root)
            # img.show()
            rows.append(data)
            img.save(save_path+num+'_.jpg')
            torchvision.utils.save_image(data, save_path+num+"_"+c_name+'.jpg', normalize=True)
        # break
    # print(len(rows))
        torchvision.utils.save_image(rows, save_path + c_name+'1111111111_.jpg', nrow=10, normalize=True)