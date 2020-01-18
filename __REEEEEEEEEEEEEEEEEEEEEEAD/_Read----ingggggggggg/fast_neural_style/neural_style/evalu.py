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

def ss(s): raise Exception(s)

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

def main():
    from argparse import Namespace
    args = Namespace()
    args.content_image = '/mnt/D8442D91442D7382/Mystuff/Workspace/python_world/python_github/__SSSSTTTTOOOORRRREEEE/1/2/WeChat Image_20190618024212.jpg'
    args.content_scale = 4
    args.output_image = '/mnt/D8442D91442D7382/Mystuff/Workspace/python_world/python_github/__SSSSTTTTOOOORRRREEEE/s/imgs/out.png'
    args.model = '/mnt/D8442D91442D7382/Mystuff/Workspace/python_world/python_github/__SSSSTTTTOOOORRRREEEE/s/epoch_2000_Thu_Jan_16_20:52:31_2020_100000.0_10000000000.0.model'
    args.is_cuda = False

    stylize(args)

if __name__ == "__main__":
    main()
