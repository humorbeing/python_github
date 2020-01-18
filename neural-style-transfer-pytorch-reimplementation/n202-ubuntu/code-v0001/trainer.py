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


def ss(s):
    import sys
    print(s)
    sys.exit(1)

def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        # if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):
        #     os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


def train(args):
    check_paths(args)


    device = torch.device("cuda" if args.is_cuda else "cpu")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    transformer = TransformerNet().to(device)
    optimizer = Adam(transformer.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16(requires_grad=False).to(device)
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    style = utils.load_image(args.style_image, size=args.style_size)
    # print(style.size)
    # ss('yo')
    style = style_transform(style)  # it's not transform
    style = style.repeat(args.batch_size, 1, 1, 1).to(device)
    # style = style.repeat(2,1,1,1).to(device)
    # print(style.shape)
    # print()
    # ss('ho')
    features_style = vgg(utils.normalize_batch(style))
    # print(features_style.relu4_3.shape)
    # for i in features_style:
    #     print(i.shape)
    # ss('normalize')
    gram_style = [utils.gram_matrix(y) for y in features_style]
    # for i in gram_style:
        # print(i.shape)
    # ss('main: gram style')
    for e in range(args.epochs):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            # print(n_batch)
            # ss('hi')
            count += n_batch
            optimizer.zero_grad()

            x = x.to(device)
            # print(x.shape)
            # print(x[0,0,0,:])
            # ss('in epoch, batch')
            y = transformer(x)
            # ss('in epoch, batch')
            y = utils.normalize_batch(y)
            x = utils.normalize_batch(x)

            features_y = vgg(y)
            features_x = vgg(x)

            content_loss = args.content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = utils.gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= args.style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()

            # if (batch_id + 1) % args.log_interval == 0:
            if True:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                  (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(mesg)
            if args.is_quickrun:
                if count > 10:
                    break
            # if args.checkpoint_model_dir is not None and (batch_id + 1) % args.checkpoint_interval == 0:
            #     transformer.eval().cpu()
            #     ckpt_model_filename = "ckpt_epoch_" + str(e) + "_batch_id_" + str(batch_id + 1) + ".pth"
            #     ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
            #     torch.save(transformer.state_dict(), ckpt_model_path)
            #     transformer.to(device).train()
            # if (e+1) % 50 == 0:
            #     # utils.save_image(args.save_model_dir+'/imgs/npepoch_{}.png'.format(e), y[0].detach().cpu())
            #     torchvision.utils.save_image(y[0], args.save_model_dir+'/imgs/epoch_{}.png'.format(e), normalize=True)

            # ss('yo')
    # save model
    transformer.eval().cpu()

    save_model_filename = "style_"+args.style_name+"_epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
        args.content_weight) + "_" + str(args.style_weight) + ".model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(transformer.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)



if __name__ == "__main__":
    pass
