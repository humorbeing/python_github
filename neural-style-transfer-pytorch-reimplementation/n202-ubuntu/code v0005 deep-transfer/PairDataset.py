import os
import torch
from log_utils import get_logger
from im_utils import load_img
from torch.utils.data import Dataset

log = get_logger()
supported_img_formats = ('.png', '.jpg', '.jpeg')

class ContentStylePairDataset(Dataset):

    def __init__(self, args):
        super(Dataset, self).__init__()

        self.contentSize = args.contentSize
        self.styleSize = args.styleSize

        self.pairs_fn = [(args.content, args.style)]


    def __len__(self):
        return len(self.pairs_fn)

    def __getitem__(self, idx):
        pair = self.pairs_fn[idx]

        style = load_img(pair[1], self.styleSize)

        content = load_img(pair[0], self.contentSize)

        return {'content': content, 'contentPath': pair[0], 'style': style, 'stylePath': pair[1]}