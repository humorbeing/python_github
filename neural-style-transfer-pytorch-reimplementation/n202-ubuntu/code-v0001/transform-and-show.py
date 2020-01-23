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

def stylize_one(style_model_path, target_image):
    content_image = utils.load_image(target_image, scale=4)
    print('content_image', content_image)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0)
    with torch.no_grad():

        style_model = TransformerNet()
        state_dict = torch.load(style_model_path)
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]

        style_model.load_state_dict(state_dict)
        output = style_model(content_image)
        data = output[0]
        # torchvision.utils.save_image(data, './1.png', normalize=True)
        img = data.clone().clamp(0, 255).numpy()
        img = img.transpose(1, 2, 0).astype("uint8")
        img = Image.fromarray(img)
    return img

def stylize(args):
    device = torch.device("cuda" if args.is_cuda else "cpu")

    content_image = utils.load_image(args.content_image, scale=args.content_scale)
    # print(content_image.size)
    # ss('stop')
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)
    print(content_image.shape)
    # ss('stop')

    with torch.no_grad():
        print(1)
        style_model = TransformerNet()
        print(2)
        state_dict = torch.load(args.model)
        print(3)
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        # for k in list(state_dict.keys()):
        #     if re.search(r'in\d+\.running_(mean|var)$', k):
        #         del state_dict[k]
        print(4)
        style_model.load_state_dict(state_dict)
        print(5)
        style_model.to(device)
        print(6)

        output = style_model(content_image).cpu()
        print(output.shape)
        # ss('s')
    utils.save_image(args.output_image, output[0])
    # torchvision.utils.save_image(output[0], args.output_image, normalize=True)


def ta(model_name, image, model_root=''):
    model_path = model_root + model_name + '.model'
    img = stylize_one(model_path, image)
    return img

if __name__ == "__main__":
    # model_name = '01'

    # model_path = '/home/ray/Desktop/Link to Mystuff/Workspace/python_world/python_github/__SSSSTTTTOOOORRRREEEE/neural-style/saved-model-here/style_05.model'
    model_root = '/home/ray/Desktop/Link to Mystuff/Workspace/python_world/python_github/__SSSSTTTTOOOORRRREEEE/neural-style/saved-model-here/trained on colab/'
    target_image_path = '/home/ray/Desktop/Link to Mystuff/Workspace/python_world/python_github/__SSSSTTTTOOOORRRREEEE/neural-style/1/WeChat Image_20200118161228.jpg'
    target_image_path = '/home/ray/Desktop/Link to Mystuff/Workspace/python_world/python_github/__SSSSTTTTOOOORRRREEEE/neural-style/all content/content/20181007_131604.jpg'
    save_path = './img/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # img = ta(model_name, target_image_path, model_root)
    # img.show()
    for i in range(1,26):
        num = '{:02d}'.format(i)
        img = ta(num, target_image_path, model_root)
        # img.show()
        img.save(save_path+num+'_.jpg')