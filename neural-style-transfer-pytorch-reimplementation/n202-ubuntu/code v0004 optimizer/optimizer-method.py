import torch
import torch.nn as nn
import torch.optim as optim
import os
import torchvision
import numpy as np
from torchvision import models, transforms
from PIL import Image

import utility


VGG19_PATH = '/home/ray/Desktop/Link to Mystuff/Workspace/python_world/python_github/__SSSSTTTTOOOORRRREEEE/neural-style/vgg19 pretrained model weights/vgg19-d01eb7cb.pth'
# Load VGG19 Skeleton
vgg = models.vgg19(pretrained=False)

# Load pretrained weights
vgg.load_state_dict(torch.load(VGG19_PATH), strict=False)
for param in vgg.parameters():
    param.requires_grad = False
# print(vgg)

for i, layer in enumerate(vgg.features):
    if isinstance(layer, nn.MaxPool2d):
        vgg.features[i] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
# print(vgg)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg.to(device).eval()

inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

def load_image(image_path, size=512):
    image = utility.open_image_ok_size(image_path, size=size)
    in_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229, 0.224, 0.225))
    ])
    image = in_transform(image).unsqueeze(0)
    return image


style_weights = {
    'conv1_1': 0.75,
    'conv2_1': 0.5,
    'conv3_1': 0.2,
    'conv4_1': 0.2,
    'conv5_1': 0.2,
}

def get_features(image, model, layers=None):
    if layers is None:
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',
            '28': 'conv5_1',
        }
    features = {}
    x = image
    for name, layer in enumerate(model.features):
        x = layer(x)
        if str(name) in layers:
            features[layers[str(name)]] = x
    return features

def gram_matrix(tensor):
    _, n_filters, h, w = tensor.size()
    tensor = tensor.view(n_filters, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram


def optimizer_method(content, style, epoch=5000, v=None, path='./imgs/name/'):
    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)

    target = torch.randn_like(content).requires_grad_(True).to(device)

    content_weight = 1e4
    style_weight = 1e2

    optimizer = optim.Adam([target], lr=0.01)
    mse_loss = torch.nn.MSELoss()

    for e in range(epoch):
        optimizer.zero_grad()
        target_features = get_features(target, vgg)

        content_loss = mse_loss(target_features['conv4_2'], content_features['conv4_2'])

        style_loss = 0
        for layer in style_weights:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            style_feature = style_features[layer]
            style_gram = gram_matrix(style_feature)
            layer_style_loss = style_weights[layer] * mse_loss(target_gram, style_gram)
            (b, c, h, w) = target_feature.size()
            style_loss += layer_style_loss / (c * h * w)

        total_loss = content_weight * content_loss + style_weight * style_loss
        total_loss.backward(retain_graph=True)
        optimizer.step()

        if v != None:
            if not os.path.exists(path):
                os.makedirs(path)
            if e % v == 0:
                y = inv_normalize(target[0])
                y = y.clamp(0, 1)
                torchvision.utils.save_image(y, path+'epoch_{:04d}.png'.format(e))
    return inv_normalize(target[0]).clamp(0, 1)


content_image = '/home/ray/Desktop/Link to Mystuff/Workspace/python_world/python_github/__SSSSTTTTOOOORRRREEEE/neural-style/1/2/WeChat Image_20190618024212.jpg'
content_image = '/home/ray/Desktop/Link to Mystuff/Workspace/python_world/python_github/__SSSSTTTTOOOORRRREEEE/neural-style/all content/content/20181007_131604.jpg'
style_image = '/home/ray/Desktop/Link to Mystuff/Workspace/python_world/python_github/__SSSSTTTTOOOORRRREEEE/neural-style/s/s.jpg'
style_image = '/home/ray/Desktop/Link to Mystuff/Workspace/python_world/python_github/neural-style-transfer-pytorch-reimplementation/style-images/style-images-here/07.jpg'
style_image = '/home/ray/Desktop/Link to Mystuff/Workspace/python_world/python_github/__SSSSTTTTOOOORRRREEEE/neural-style/all style/style/tiger.jpg'
content = load_image(content_image).to(device)
style = load_image(style_image).to(device)


output = optimizer_method(content, style, v=30)

# convert -delay 1 -loop 0 *.png ../2.gif