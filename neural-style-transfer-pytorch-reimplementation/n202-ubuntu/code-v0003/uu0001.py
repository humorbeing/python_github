import torch
import torch.nn as nn
import torch.optim as optim


import numpy as np
# import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image

import copy

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

def load_image(img_path, scale=1):

    image = Image.open(img_path).convert('RGB')
    # image = image.resize((int(image.size[0] / scale), int(image.size[1] / scale)), Image.ANTIALIAS)
    in_transform = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229, 0.224, 0.225))
    ])
    image = in_transform(image).unsqueeze(0)
    return image

contant_image = '/home/ray/Desktop/Link to Mystuff/Workspace/python_world/python_github/__SSSSTTTTOOOORRRREEEE/neural-style/1/2/WeChat Image_20190618024212.jpg'
style_image = '/home/ray/Desktop/Link to Mystuff/Workspace/python_world/python_github/__SSSSTTTTOOOORRRREEEE/neural-style/s/s.jpg'
style_image = '/home/ray/Desktop/Link to Mystuff/Workspace/python_world/python_github/neural-style-transfer-pytorch-reimplementation/style-images/style-images-here/07.jpg'
content = load_image(contant_image, scale=4).to(device)
style = load_image(style_image).to(device)

# layers = {
#     '3': 'relu1_2',   # Style layers
#     '8': 'relu2_2',
#     '17' : 'relu3_3',
#     '26' : 'relu4_3',
#     '35' : 'relu5_3',
#     '22' : 'relu4_2', # Content layers
#     # '31' : 'relu5_2'
# }

# layers = {
#             '0': 'conv1_1',
#             '5': 'conv2_1',
#             '10': 'conv3_1',
#             '19': 'conv4_1',
#             '21': 'conv4_2',
#             '28': 'conv5_1',
#         }
# style_weights = {
#     'relu1_2': 1,   # Style layers
#     'relu2_2': 1,
#     'relu3_3': 2,
#     'relu4_3': 3,
#     'relu5_3': 1,
#     # 'relu5_2': 1,
#     # 'relu1_2': 1,
# }

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
# def gram_matrix(y):
#     # print(y.shape)
#     # print(y.size())
#
#     (b, ch, h, w) = y.size()
#     features = y.view(b, ch, w * h)
#
#     features_t = features.transpose(1, 2)
#     # print(features.shape)
#     # print(features_t.shape)
#     gram = features.bmm(features_t) / (ch * h * w)
#     # print(ch*h*w)
#     # ss('in gram matrix')
#     return gram
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

style_grams = {
    layer: gram_matrix(style_features[layer]) for layer in style_features
}

target = torch.randn_like(content).requires_grad_(True).to(device)



content_weight = 1e4
style_weight = 1e4

optimizer = optim.Adam([target], lr=0.01)
mse_loss = torch.nn.MSELoss()

for e in range(3000):
    optimizer.zero_grad()
    target_features = get_features(target, vgg)

    content_loss = mse_loss(target_features['conv4_2'], content_features['conv4_2'])
    # content_loss = mse_loss(target_features['relu4_2'], content_features['relu4_2'])

    style_loss = 0
    for layer in style_weights:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        style_feature = style_features[layer]
        style_gram = gram_matrix(style_feature)
        layer_style_loss = style_weights[layer] * mse_loss (target_gram, style_gram)
        (b,c,h,w) = target_feature.size()
        style_loss += layer_style_loss/ (c*h*w)

    total_loss = content_weight * content_loss + style_weight * style_loss
    total_loss.backward(retain_graph=True)
    optimizer.step()

    if e % 5 == 0:
        print('epoch is {}'.format(e))
        img = target.to('cpu').clone().detach().numpy().squeeze()
        img = img.transpose(1, 2, 0)
        img = img * np.array((0.229, 0.224, 0.225)) + np.array((0.485,0.456,0.406))
        img = img.clip(0, 1)
        img = (img * 255).astype("uint8")
        img = Image.fromarray(img)
        img.save('./imgs/epoch_{}.png'.format(e))

# img.show()