import PIL
import torch
from PIL import Image
import torch.nn as nn
from log_utils import get_logger
from feature_transforms import wct, wct_mask
from encoder_decoder_factory import Encoder, Decoder
import torchvision.transforms.functional as transforms


log = get_logger()

def ss(s):
    print(s)
    import sys
    sys.exit(1)

def stylize(level, content, style0, encoders, decoders, alpha, svd_device, cnn_device, interpolation_beta=None, style1=None, mask_mode=None, mask=None):
    # log.debug('Stylization up to ReLu' + str(level) + ' of content sized: ' + str(content.size()) + ' and style sized: ' + str(style0.size()))

    with torch.no_grad():
        # print('level', level)
        # print('c in',content.shape)
        # print('s in',style0.shape)
        # ss('in stylize')
        cf = encoders[level](content).data.to(device=svd_device).squeeze(0)
        # print(cf.shape)

        s0f = encoders[level](style0).data.to(device=svd_device).squeeze(0)
        # log.debug('transfer-mode: content features size: ' + str(cf.size()) + ', style features size: ' + str(s0f.size()))
        # print(cf.shape)
        csf = wct(alpha, cf, s0f).to(device=cnn_device)
        # print(csf.shape)

        return decoders[level](csf)

class SingleLevelWCT(nn.Module):

    def __init__(self, args):
        super(SingleLevelWCT, self).__init__()
        # ss('in singlelevelwct init')
        self.svd_device = torch.device('cpu')  # on average svd takes 4604ms on cpu vs gpu 5312ms on a 512x512 content/591x800 style (comprehensive of data transferring)
        self.cnn_device = args.device
        self.alpha = args.alpha
        self.beta = args.beta
        # ss('in singlelevelwct init')

        self.mask_mode = False
        self.mask = None


        self.e5 = Encoder(5)
        self.encoders = [self.e5]
        self.d5 = Decoder(5)
        self.decoders = [self.d5]
        # ss('in singlelevelwct init')
    def forward(self, content_img, style_img, additional_style_flag=False, style_img1=None):


        out = stylize(0, content_img, style_img, self.encoders, self.decoders, self.alpha, self.svd_device,
                      self.cnn_device)
        # print(out.shape)
        # ss('in forward')
        return out


class MultiLevelWCT(nn.Module):

    def __init__(self, args):
        super(MultiLevelWCT, self).__init__()

        self.svd_device = torch.device('cpu')
        self.cnn_device = args.device
        self.alpha = args.alpha
        self.beta = args.beta


        self.mask_mode = False
        self.mask = None

        self.e1 = Encoder(1)
        self.e2 = Encoder(2)
        self.e3 = Encoder(3)
        self.e4 = Encoder(4)
        self.e5 = Encoder(5)
        self.encoders = [self.e5, self.e4, self.e3, self.e2, self.e1]

        self.d1 = Decoder(1)
        self.d2 = Decoder(2)
        self.d3 = Decoder(3)
        self.d4 = Decoder(4)
        self.d5 = Decoder(5)
        self.decoders = [self.d5, self.d4, self.d3, self.d2, self.d1]

    def forward(self, content_img, style_img, additional_style_flag=False, style_img1=None):

        for i in range(len(self.encoders)):

            content_img = stylize(i, content_img, style_img, self.encoders, self.decoders, self.alpha, self.svd_device,
                                      self.cnn_device)
        # ss('in forward')
        return content_img