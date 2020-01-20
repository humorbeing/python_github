"""
Copyright (c) 2019 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import os
import tqdm
import argparse

import torch
from torchvision.utils import save_image

from model import WaveEncoder, WaveDecoder

from utils.core import feature_wct
from utils.io import Timer, open_image, load_segment, compute_label_info


def ss(s):
    print(s)
    import sys
    sys.exit(1)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class WCT2:
    def __init__(self, model_path='./model_checkpoints', transfer_at=['encoder', 'skip', 'decoder'], option_unpool='cat5', device='cuda:0', verbose=False):
        # ss('in wct2 init')
        self.transfer_at = set(transfer_at)


        self.device = torch.device(device)
        self.verbose = verbose
        self.encoder = WaveEncoder(option_unpool).to(self.device)
        self.decoder = WaveDecoder(option_unpool).to(self.device)
        self.encoder.load_state_dict(torch.load(os.path.join(model_path, 'wave_encoder_{}_l4.pth'.format(option_unpool)), map_location=lambda storage, loc: storage))
        self.decoder.load_state_dict(torch.load(os.path.join(model_path, 'wave_decoder_{}_l4.pth'.format(option_unpool)), map_location=lambda storage, loc: storage))

    def print_(self, msg):
        if self.verbose:
            print(msg)

    def encode(self, x, skips, level):
        return self.encoder.encode(x, skips, level)

    def decode(self, x, skips, level):
        return self.decoder.decode(x, skips, level)

    def get_all_feature(self, x):
        skips = {}
        feats = {'encoder': {}, 'decoder': {}}
        for level in [1, 2, 3, 4]:
            x = self.encode(x, skips, level)
            if 'encoder' in self.transfer_at:
                feats['encoder'][level] = x

        if 'encoder' not in self.transfer_at:
            feats['decoder'][4] = x
        for level in [4, 3, 2]:
            x = self.decode(x, skips, level)
            if 'decoder' in self.transfer_at:
                feats['decoder'][level - 1] = x
        return feats, skips

    def transfer(self, content, style, content_segment, style_segment, alpha=1):

        label_set, label_indicator = None, None
        content_feat, content_skips = content, {}
        style_feats, style_skips = self.get_all_feature(style)
        # print(content_feat.shape)
        # print(style_feats['encoder'][1].shape)
        wct2_enc_level = [1, 2, 3, 4]
        wct2_dec_level = [1, 2, 3, 4]
        wct2_skip_level = ['pool1', 'pool2', 'pool3']

        for level in [1, 2, 3, 4]:
            content_feat = self.encode(content_feat, content_skips, level)
            if 'encoder' in self.transfer_at and level in wct2_enc_level:
                content_feat = feature_wct(content_feat, style_feats['encoder'][level],
                                           content_segment, style_segment,
                                           label_set, label_indicator,
                                           alpha=alpha, device=self.device)
                #
                self.print_('transfer at encoder {}'.format(level))
        if 'skip' in self.transfer_at:
            for skip_level in wct2_skip_level:
                for component in [0, 1, 2]:  # component: [LH, HL, HH]
                    content_skips[skip_level][component] = feature_wct(content_skips[skip_level][component], style_skips[skip_level][component],
                                                                       content_segment, style_segment,
                                                                       label_set, label_indicator,
                                                                       alpha=alpha, device=self.device)
                self.print_('transfer at skip {}'.format(skip_level))

        for level in [4, 3, 2, 1]:
            if 'decoder' in self.transfer_at and level in style_feats['decoder'] and level in wct2_dec_level:
                content_feat = feature_wct(content_feat, style_feats['decoder'][level],
                                           content_segment, style_segment,
                                           label_set, label_indicator,
                                           alpha=alpha, device=self.device)
                self.print_('transfer at decoder {}'.format(level))
            content_feat = self.decode(content_feat, content_skips, level)
        # print(content_feat.shape)
        # ss('in wct2 transfer')
        return content_feat


def get_all_transfer():
    ret = []
    for e in ['encoder', None]:
        for d in ['decoder', None]:
            for s in ['skip', None]:
                _ret = set([e, d, s]) & set(['encoder', 'decoder', 'skip'])
                if _ret:
                    ret.append(_ret)
    return ret


def run_bulk(_content, s, args, root=''):
    device = 'cuda:0' if args.is_cuda else 'cpu'
    device = torch.device(device)



    # The filenames of the content and style pair should match
    # fnames = set(os.listdir(args.content)) & set(os.listdir(args.style))
    # print(len(fnames))


    # for fname in fnames:


    # _content = os.path.join(args.content, fname)
    _style = root + s +'.jpg'
    # print(_content)
    # print(_style)
    # ss('s')
    content = open_image(_content, args.image_size).to(device)
    style = open_image(_style, args.image_size).to(device)
    # content_segment = load_segment(_content_segment, config.image_size)
    # style_segment = load_segment(_style_segment, config.image_size)
    # print(style.shape)
    # print(get_all_transfer())
    for _transfer_at in get_all_transfer():
        # print(_transfer_at)
        # with Timer('Elapsed time in whole WCT: {}', config.verbose):
        postfix = '_'.join(sorted(list(_transfer_at)))
        # print(postfix)
        # fname_output = _output.replace('.png', '_{}_{}.png'.format(config.option_unpool, postfix))
        # print('------ transfer:', fname)
        #
        wct2 = WCT2(transfer_at=_transfer_at, option_unpool=args.option_unpool, device=device)

        with torch.no_grad():
            img = wct2.transfer(content, style, None, None, alpha=args.alpha)
        # ss('in run bulk')
        import torchvision
        img.clamp(0, 1)
        # torchvision.utils.save_image(content, './imgs/c.png')  # , normalize=True)
        # torchvision.utils.save_image(style, './imgs/s.png')  # , normalize=True)
        torchvision.utils.save_image(img, './img/'+s+postfix+'_t.png')  # , normalize=True)
        # save_image(img.clamp_(0, 1), fname_output, padding=0)
        # break


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--content', type=str, default='./examples/content')
    # parser.add_argument('--content_segment', type=str, default=None)
    # parser.add_argument('--style', type=str, default='./examples/style')
    # parser.add_argument('--style_segment', type=str, default=None)
    # parser.add_argument('--output', type=str, default='./outputs')
    # parser.add_argument('--image_size', type=int, default=512)
    # parser.add_argument('--alpha', type=float, default=1)
    # parser.add_argument('--option_unpool', type=str, default='cat5', choices=['sum', 'cat5'])
    # parser.add_argument('-e', '--transfer_at_encoder', action='store_true')
    # parser.add_argument('-d', '--transfer_at_decoder', action='store_true')
    # parser.add_argument('-s', '--transfer_at_skip', action='store_true')
    # parser.add_argument('-a', '--transfer_all', action='store_true')
    # parser.add_argument('--cpu', action='store_true')
    # parser.add_argument('--verbose', action='store_true')
    # config = parser.parse_args()
    #
    # print(config)
    args = argparse.Namespace()
    args.is_cuda = True
    # args.content = 'examples/content'
    # args.style = 'examples/style'
    args.image_size = 512
    args.option_unpool = 'cat5'
    args.alpha = 1
    # if not os.path.exists(os.path.join(config.output)):
    #     os.makedirs(os.path.join(config.output))
    s ='01'

    root = '/home/ray/Desktop/Link to Mystuff/Workspace/python_world/python_github/__SSSSTTTTOOOORRRREEEE/neural-style/all style/style/'
    '''
    CUDA_VISIBLE_DEVICES=6 python transfer.py --content ./examples/content --style ./examples/style --content_segment ./examples/content_segment --style_segment ./examples/style_segment/ --output ./outputs/ --verbose --image_size 512 -a
    '''
    content_path = '/home/ray/Desktop/Link to Mystuff/Workspace/python_world/python_github/__SSSSTTTTOOOORRRREEEE/neural-style/all content/content/20181007_131604.jpg'
    run_bulk(content_path, s, args, root)
    for i in range(1, 27):
        s = '{:02d}'.format(i)
        run_bulk(content_path, s, args, root)