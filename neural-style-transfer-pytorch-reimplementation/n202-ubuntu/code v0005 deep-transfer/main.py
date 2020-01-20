import os
import re
import torch
import argparse
import PairDataset
import TripletDataset
import torchvision
import autoencoder
from log_utils import get_logger
# from torch.utils.data import DataLoaderf
from torchvision import transforms
from PIL import Image

def ss(s):
    print(s)
    import sys
    sys.exit(1)

log = get_logger()

def open_image_ok_size(image_path, size=512):
    image = Image.open(image_path)
    image = transforms.Resize(size)(image)
    w, h = image.size
    image = transforms.CenterCrop(((h // 16) * 16, (w // 16) * 16))(image)
    return image
def load_image(image_path, size=512):
    image = open_image_ok_size(image_path, size=size)
    return transforms.ToTensor()(image).unsqueeze(0)

def parse_args():
    parser = argparse.ArgumentParser(description='Pytorch implementation of arbitrary style transfer via CNN features WCT trasform',
                                     epilog='Supported image file formats are: jpg, jpeg, png')

    parser.add_argument('--content', help='Path of the content image (or a directory containing images) to be trasformed')
    parser.add_argument('--style', help='Path of the style image (or a directory containing images) to use')
    parser.add_argument('--synthesis', default=False, action='store_true', help='Flag to syntesize a new texture. Must provide a texture style image')
    parser.add_argument('--stylePair', help='Path of two style images (separated by ",") to use in combination')
    parser.add_argument('--mask', help='Path of the binary mask image (white on black) to trasfer the style pair in the corrisponding areas')

    parser.add_argument('--contentSize', type=int, help='Reshape content image to have the new specified maximum size (keeping aspect ratio)') # default=768 in the paper
    parser.add_argument('--styleSize', type=int, help='Reshape style image to have the new specified maximum size (keeping aspect ratio)')

    parser.add_argument('--outDir', default='outputs', help='Path of the directory where stylized results will be saved')
    parser.add_argument('--outPrefix', help='Name prefixed in the saved stylized images')

    parser.add_argument('--alpha', type=float, default=0.2, help='Hyperparameter balancing the blending between original content features and WCT-transformed features')
    parser.add_argument('--beta', type=float, default=0.5, help='Hyperparameter balancing the interpolation between the two images in the stylePair')
    parser.add_argument('--no-cuda', default=False, action='store_true', help='Flag to enables GPU (CUDA) accelerated computations')
    parser.add_argument('--single-level', default=False, action='store_true', help='Flag to switch to single level stylization')

    return parser.parse_args()





def save_image(img, content_name, style_name, out_ext, args):
    torchvision.utils.save_image(img.cpu().detach().squeeze(0),
     os.path.join(args.outDir,
      (args.outPrefix + '_' if args.outPrefix else '') + content_name + '_stylized_by_' + style_name + '_alpha_' + str(int(args.alpha*100)) + '.' + out_ext))


def main(c,s,out_path,args,root=''):
    style_image = root + s + '.jpg'
    model = autoencoder.MultiLevelWCT(args)
    model.to(device)
    model.eval()
    content = load_image(c).to(device=args.device)
    # print(content.shape)
    style = load_image(style_image).to(device=args.device)
    out = model(content, style)
    # print(out.shape)
    out.clamp(0,1)
    torchvision.utils.save_image(out, out_path+s+'_t.png')




if __name__ == "__main__":
    args = argparse.Namespace()
    args.outDir = 'new output'
    args.is_cuda = False
    device = torch.device('cuda:0' if args.is_cuda else 'cpu')
    args.device = device
    args.alpha = 0.5  # 1 means all style
    args.beta = 0.5


    args.contentSize = 512
    args.styleSize = 512
    s = '02'
    root = 'inputs/styles/'
    root = '/home/ray/Desktop/Link to Mystuff/Workspace/python_world/python_github/__SSSSTTTTOOOORRRREEEE/neural-style/all style/style/'
    out_path = './img/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    content = 'inputs/contents/face.jpg'
    content = '/home/ray/Desktop/Link to Mystuff/Workspace/python_world/python_github/__SSSSTTTTOOOORRRREEEE/neural-style/all content/content/20181007_131604.jpg'
    # main(content, s, out_path, args, root)
    for i in range(1, 27):
        num = '{:02d}'.format(i)
        main(content, num, out_path, args, root)