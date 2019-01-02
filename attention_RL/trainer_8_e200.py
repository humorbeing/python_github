import torch
import torch.nn.functional as F

from torch.autograd import Variable
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os
import time
import shutil
import pickle

from tqdm import tqdm
from utils import AverageMeter
from model import RecurrentAttention
# from tensorboard_logger import configure, log_value

# print(RecurrentAttention)
class Trainer(object):
    """
    Trainer encapsulates all the logic necessary for
    training the Recurrent Attention Model.

    All hyperparameters are provided by the user in the
    config file.
    """
    def __init__(self, data_loader):
        """
        Construct a new Trainer instance.

        Args
        ----
        - config: object containing command line arguments.
        - data_loader: data iterator
        """
        # self.config = config

        # glimpse network params
        self.patch_size = 8
        self.glimpse_scale = 2
        self.num_patches = 3
        self.loc_hidden = 128
        self.glimpse_hidden = 128

        # core network params
        self.num_glimpses = 6
        self.hidden_size = 256

        # reinforce params
        self.std = 0.17
        self.M = 10

        # data params

        self.train_loader = data_loader[0]
        self.valid_loader = data_loader[1]
        self.num_train = len(self.train_loader.sampler.indices)
        self.num_valid = len(self.valid_loader.sampler.indices)


        self.num_classes = 27
        self.num_channels = 3

        # training params
        self.epochs = 200
        self.start_epoch = 0
        self.saturate_epoch = 150
        self.init_lr = 0.001
        self.min_lr = 1e-06
        self.decay_rate = (self.min_lr - self.init_lr) / (self.saturate_epoch)
        self.momentum = 0.5
        self.lr = self.init_lr

        # misc params
        self.use_gpu = False
        self.best = True
        # self.ckpt_dir = config.ckpt_dir
        # self.logs_dir = config.logs_dir
        self.best_valid_acc = 0.
        self.counter = 0
        # self.patience = config.patience
        # self.use_tensorboard = config.use_tensorboard
        # self.resume = config.resume
        # self.print_freq = config.print_freq
        # self.plot_freq = config.plot_freq


        # self.plot_dir = './plots/' + self.model_name + '/'
        # if not os.path.exists(self.plot_dir):
        #     os.makedirs(self.plot_dir)

        # configure tensorboard logging


        # build RAM model
        self.model = RecurrentAttention(
            self.patch_size, self.num_patches, self.glimpse_scale,
            self.num_channels, self.loc_hidden, self.glimpse_hidden,
            self.std, self.hidden_size, self.num_classes,
        )
        if self.use_gpu:
            self.model.cuda()

        print('[*] Number of model parameters: {:,}'.format(
            sum([p.data.nelement() for p in self.model.parameters()])))

        # initialize optimizer and scheduler
        self.optimizer = SGD(
            self.model.parameters(), lr=self.lr, momentum=self.momentum,
        )
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')

    def reset(self):
        """
        Initialize the hidden state of the core network
        and the location vector.

        This is called once every time a new minibatch
        `x` is introduced.
        """
        h_t = torch.zeros(self.batch_size, self.hidden_size)
        h_t = Variable(h_t)

        l_t = torch.Tensor(self.batch_size, 2).uniform_(-1, 1)
        l_t = Variable(l_t)

        return h_t, l_t

    def train(self):
        """
        Train the model on the training set.

        A checkpoint of the model is saved after each epoch
        and if the validation accuracy is improved upon,
        a separate ckpt is created for use on the test set.
        """
        # load the most recent checkpoint
        # if self.resume:
        #     self.load_checkpoint(best=False)

        print("\n[*] Train on {} samples, validate on {} samples".format(
            self.num_train, self.num_valid)
        )

        for epoch in range(self.epochs):

            print(
                '\nEpoch: {}/{} - LR: {:.6f}'.format(
                    epoch+1, self.epochs, self.lr)
            )

            # train for 1 epoch
            train_loss, train_acc = self.train_one_epoch(epoch)

            # evaluate on validation set
            valid_loss, valid_acc = self.validate(epoch)

            # self.scheduler.step(valid_loss)
            #
            # # # decay learning rate
            # # if epoch < self.saturate_epoch:
            # #     self.anneal_learning_rate(epoch)
            #
            is_best = valid_acc > self.best_valid_acc

            msg1 = "train loss: {:.3f} - train acc: {:.3f} "
            msg2 = "- val loss: {:.3f} - val acc: {:.3f}"
            if is_best:
                msg2 += " [*]"
            msg = msg1 + msg2
            print(msg.format(train_loss, train_acc, valid_loss, valid_acc))

            # # check for improvement
            # if not is_best:
            #     self.counter += 1
            # if self.counter > self.patience:
            #     print("[!] No improvement in a while, stopping training.")
            #     return
            # self.best_valid_acc = max(valid_acc, self.best_valid_acc)
            # self.save_checkpoint(
            #     {'epoch': epoch + 1, 'state_dict': self.model.state_dict(),
            #      'best_valid_acc': self.best_valid_acc,
            #      'lr': self.lr}, is_best
            # )

    def train_one_epoch(self, epoch):
        """
        Train the model for 1 epoch of the training set.

        An epoch corresponds to one full pass through the entire
        training set in successive mini-batches.

        This is used by train() and should not be called manually.
        """
        batch_time = AverageMeter()
        losses = AverageMeter()
        accs = AverageMeter()

        tic = time.time()

        for i, (x, y) in enumerate(self.train_loader):
            if self.use_gpu:
                x, y = x.cuda(), y.cuda()
            x, y = Variable(x), Variable(y)

            # plot = False
            # if (epoch % self.plot_freq == 0) and (i == 0):
            #     plot = True

            # initialize location vector and hidden state
            self.batch_size = x.shape[0]
            h_t, l_t = self.reset()

            # save images
            # imgs = []
            # imgs.append(x[0:9])

            # extract the glimpses
            locs = []
            log_pi = []
            baselines = []
            for t in range(self.num_glimpses - 1):

                # forward pass through model
                h_t, l_t, b_t, p = self.model(x, l_t, h_t)

                # store
                locs.append(l_t[0:9])
                baselines.append(b_t)
                log_pi.append(p)

            # last iteration
            h_t, l_t, b_t, log_probas, p = self.model(
                x, l_t, h_t, last=True
            )
            log_pi.append(p)
            baselines.append(b_t)
            # locs.append(l_t[0:9])

            # convert list to tensors and reshape
            baselines = torch.stack(baselines).transpose(1, 0)
            log_pi = torch.stack(log_pi).transpose(1, 0)

            # calculate reward
            predicted = torch.max(log_probas, 1)[1]
            R = (predicted.detach() == y).float()
            R = R.unsqueeze(1).repeat(1, self.num_glimpses)

            # compute losses for differentiable modules
            loss_action = F.nll_loss(log_probas, y)
            loss_baseline = F.mse_loss(baselines, R)

            # compute reinforce loss
            adjusted_reward = R - baselines.detach()
            loss_reinforce = torch.mean(-log_pi*adjusted_reward)

            # sum up into a hybrid loss
            loss = loss_action + loss_baseline + loss_reinforce

            # compute accuracy
            correct = (predicted == y).float()
            acc = 100 * (correct.sum() / len(y))

            # store
            losses.update(loss.item(), x.size()[0])
            accs.update(acc.item(), x.size()[0])

            # a = list(self.model.sensor.parameters())[0].clone()
            # self.optimizer.zero_grad()
            # loss_reinforce.backward()
            # self.optimizer.step()
            # b = list(self.model.sensor.parameters())[0].clone()
            # print("Same: {}".format(torch.equal(a.data, b.data)))

            # compute gradients and update SGD
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            toc = time.time()
            batch_time.update(toc-tic)

            # print("{:.1f}s - loss: {:.3f} - acc: {:.3f}".format(
            #             (toc-tic), loss.data[0], acc.data[0]
            #             ))




        return losses.avg, accs.avg

    def validate(self, epoch):
        """
        Evaluate the model on the validation set.
        """
        losses = AverageMeter()
        accs = AverageMeter()

        for i, (x, y) in enumerate(self.valid_loader):
            if self.use_gpu:
                x, y = x.cuda(), y.cuda()
            x, y = Variable(x), Variable(y)

            # duplicate 10 times
            x = x.repeat(self.M, 1, 1, 1)

            # initialize location vector and hidden state
            self.batch_size = x.shape[0]
            h_t, l_t = self.reset()

            # extract the glimpses
            log_pi = []
            baselines = []
            for t in range(self.num_glimpses - 1):

                # forward pass through model
                h_t, l_t, b_t, p = self.model(x, l_t, h_t)

                # store
                baselines.append(b_t)
                log_pi.append(p)

            # last iteration
            h_t, l_t, b_t, log_probas, p = self.model(
                x, l_t, h_t, last=True
            )
            log_pi.append(p)
            baselines.append(b_t)

            # convert list to tensors and reshape
            baselines = torch.stack(baselines).transpose(1, 0)
            log_pi = torch.stack(log_pi).transpose(1, 0)

            # average
            log_probas = log_probas.view(
                self.M, -1, log_probas.shape[-1]
            )
            log_probas = torch.mean(log_probas, dim=0)

            baselines = baselines.contiguous().view(
                self.M, -1, baselines.shape[-1]
            )
            baselines = torch.mean(baselines, dim=0)

            log_pi = log_pi.contiguous().view(
                self.M, -1, log_pi.shape[-1]
            )
            log_pi = torch.mean(log_pi, dim=0)

            # calculate reward
            predicted = torch.max(log_probas, 1)[1]
            R = (predicted.detach() == y).float()
            R = R.unsqueeze(1).repeat(1, self.num_glimpses)

            # compute losses for differentiable modules
            loss_action = F.nll_loss(log_probas, y)
            loss_baseline = F.mse_loss(baselines, R)

            # compute reinforce loss
            adjusted_reward = R - baselines.detach()
            loss_reinforce = torch.mean(-log_pi*adjusted_reward)

            # sum up into a hybrid loss
            loss = loss_action + loss_baseline + loss_reinforce

            # compute accuracy
            correct = (predicted == y).float()
            acc = 100 * (correct.sum() / len(y))

            # store
            losses.update(loss.item(), x.size()[0])
            accs.update(acc.item(), x.size()[0])


        return losses.avg, accs.avg

    def test(self, loader):
        """
        Test the model on the held-out test data.
        This function should only be called at the very
        end once the model has finished training.
        """
        correct = 0
        self.test_loader = loader
        # load the best checkpoint
        # self.load_checkpoint(best=self.best)
        self.num_test = len(self.test_loader.dataset)

        for i, (x, y) in enumerate(self.test_loader):
            # if self.use_gpu:
            #     x, y = x.cuda(), y.cuda()
            x, y = Variable(x), Variable(y)

            # duplicate 10 times
            x = x.repeat(self.M, 1, 1, 1)

            # initialize location vector and hidden state
            self.batch_size = x.shape[0]
            h_t, l_t = self.reset()

            # extract the glimpses
            for t in range(self.num_glimpses - 1):

                # forward pass through model
                h_t, l_t, b_t, p = self.model(x, l_t, h_t)

            # last iteration
            h_t, l_t, b_t, log_probas, p = self.model(
                x, l_t, h_t, last=True
            )

            log_probas = log_probas.view(
                self.M, -1, log_probas.shape[-1]
            )
            log_probas = torch.mean(log_probas, dim=0)

            pred = log_probas.data.max(1, keepdim=True)[1]
            correct += pred.eq(y.data.view_as(pred)).cpu().sum()

        perc = (100. * correct) / (self.num_test)
        print(
            '[*] Test Acc: {}/{} ({:.2f}%)'.format(
                correct, self.num_test, perc)
        )

    def anneal_learning_rate(self, epoch):
        """
        This function linearly decays the learning rate
        to a predefined minimum over a set amount of epochs.
        """
        self.lr += self.decay_rate

        # log to tensorboard
        if self.use_tensorboard:
            log_value('learning_rate', self.lr, epoch)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def save_checkpoint(self, state, is_best):
        """
        Save a copy of the model so that it can be loaded at a future
        date. This function is used when the model is being evaluated
        on the test data.

        If this model has reached the best validation accuracy thus
        far, a seperate file with the suffix `best` is created.
        """
        # print("[*] Saving model to {}".format(self.ckpt_dir))

        filename = self.model_name + '_ckpt.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)

        if is_best:
            filename = self.model_name + '_model_best.pth.tar'
            shutil.copyfile(
                ckpt_path, os.path.join(self.ckpt_dir, filename)
            )

    def load_checkpoint(self, best=False):
        """
        Load the best copy of a model. This is useful for 2 cases:

        - Resuming training with the most recent model checkpoint.
        - Loading the best validation model to evaluate on the test data.

        Params
        ------
        - best: if set to True, loads the best model. Use this if you want
          to evaluate your model on the test data. Else, set to False in
          which case the most recent version of the checkpoint is used.
        """
        print("[*] Loading model from {}".format(self.ckpt_dir))

        filename = self.model_name + '_ckpt.pth.tar'
        if best:
            filename = self.model_name + '_model_best.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        ckpt = torch.load(ckpt_path)

        # load variables from checkpoint
        self.start_epoch = ckpt['epoch']
        self.best_valid_acc = ckpt['best_valid_acc']
        self.lr = ckpt['lr']
        self.model.load_state_dict(ckpt['state_dict'])

        if best:
            print(
                "[*] Loaded {} checkpoint @ epoch {} "
                "with best valid acc of {:.3f}".format(
                    filename, ckpt['epoch']+1, ckpt['best_valid_acc'])
            )
        else:
            print(
                "[*] Loaded {} checkpoint @ epoch {}".format(
                    filename, ckpt['epoch']+1)
            )


# from trainer import Trainer
from dataloader import train_set_test_set
from dataloader import test_set

dataset_path = '/media/ray/SSD/workspace/python/dataset/trainset_testset/fish_dataset_train_test_split/train'
d, s, c = train_set_test_set(
    dataset_path, 64
)
t_p = '/media/ray/SSD/workspace/python/dataset/trainset_testset/fish_dataset_train_test_split/test'
t_set = test_set(t_p)
trainer = Trainer(d)
trainer.train()
trainer.test(t_set)

'''/media/ray/SSD/workspace/python/ENVIRONMENT/python3.5/bin/python3.5 /media/ray/SSD/workspace/python/attention_RL/recurrent-visual-attention/trainer_8_e200.py
These is no random seed given.
[*] Number of model parameters: 279,582

[*] Train on 1085 samples, validate on 465 samples

Epoch: 1/200 - LR: 0.001000
train loss: 3.306 - train acc: 5.991 - val loss: 3.303 - val acc: 6.882 [*]

Epoch: 2/200 - LR: 0.001000
train loss: 3.307 - train acc: 7.373 - val loss: 3.302 - val acc: 7.097 [*]

Epoch: 3/200 - LR: 0.001000
train loss: 3.302 - train acc: 7.742 - val loss: 3.303 - val acc: 7.527 [*]

Epoch: 4/200 - LR: 0.001000
train loss: 3.305 - train acc: 7.926 - val loss: 3.301 - val acc: 7.312 [*]

Epoch: 5/200 - LR: 0.001000
train loss: 3.300 - train acc: 7.834 - val loss: 3.297 - val acc: 6.882 [*]

Epoch: 6/200 - LR: 0.001000
train loss: 3.294 - train acc: 7.097 - val loss: 3.297 - val acc: 7.097 [*]

Epoch: 7/200 - LR: 0.001000
train loss: 3.294 - train acc: 7.281 - val loss: 3.293 - val acc: 6.882 [*]

Epoch: 8/200 - LR: 0.001000
train loss: 3.298 - train acc: 7.558 - val loss: 3.296 - val acc: 6.882 [*]

Epoch: 9/200 - LR: 0.001000
train loss: 3.296 - train acc: 8.018 - val loss: 3.293 - val acc: 6.667 [*]

Epoch: 10/200 - LR: 0.001000
train loss: 3.291 - train acc: 6.912 - val loss: 3.288 - val acc: 6.882 [*]

Epoch: 11/200 - LR: 0.001000
train loss: 3.290 - train acc: 7.742 - val loss: 3.287 - val acc: 6.882 [*]

Epoch: 12/200 - LR: 0.001000
train loss: 3.288 - train acc: 7.742 - val loss: 3.283 - val acc: 6.882 [*]

Epoch: 13/200 - LR: 0.001000
train loss: 3.294 - train acc: 8.018 - val loss: 3.285 - val acc: 6.882 [*]

Epoch: 14/200 - LR: 0.001000
train loss: 3.285 - train acc: 7.834 - val loss: 3.283 - val acc: 6.667 [*]

Epoch: 15/200 - LR: 0.001000
train loss: 3.285 - train acc: 7.373 - val loss: 3.282 - val acc: 6.667 [*]

Epoch: 16/200 - LR: 0.001000
train loss: 3.279 - train acc: 7.650 - val loss: 3.281 - val acc: 6.667 [*]

Epoch: 17/200 - LR: 0.001000
train loss: 3.280 - train acc: 7.373 - val loss: 3.280 - val acc: 6.667 [*]

Epoch: 18/200 - LR: 0.001000
train loss: 3.279 - train acc: 7.558 - val loss: 3.278 - val acc: 6.667 [*]

Epoch: 19/200 - LR: 0.001000
train loss: 3.278 - train acc: 7.742 - val loss: 3.276 - val acc: 6.667 [*]

Epoch: 20/200 - LR: 0.001000
train loss: 3.281 - train acc: 7.465 - val loss: 3.275 - val acc: 6.667 [*]

Epoch: 21/200 - LR: 0.001000
train loss: 3.271 - train acc: 7.373 - val loss: 3.272 - val acc: 6.667 [*]

Epoch: 22/200 - LR: 0.001000
train loss: 3.272 - train acc: 7.742 - val loss: 3.270 - val acc: 6.667 [*]

Epoch: 23/200 - LR: 0.001000
train loss: 3.271 - train acc: 7.834 - val loss: 3.270 - val acc: 6.667 [*]

Epoch: 24/200 - LR: 0.001000
train loss: 3.269 - train acc: 7.742 - val loss: 3.267 - val acc: 6.667 [*]

Epoch: 25/200 - LR: 0.001000
train loss: 3.264 - train acc: 7.742 - val loss: 3.268 - val acc: 6.667 [*]

Epoch: 26/200 - LR: 0.001000
train loss: 3.267 - train acc: 7.742 - val loss: 3.262 - val acc: 6.667 [*]

Epoch: 27/200 - LR: 0.001000
train loss: 3.268 - train acc: 7.742 - val loss: 3.261 - val acc: 6.667 [*]

Epoch: 28/200 - LR: 0.001000
train loss: 3.258 - train acc: 7.742 - val loss: 3.261 - val acc: 6.667 [*]

Epoch: 29/200 - LR: 0.001000
train loss: 3.268 - train acc: 7.742 - val loss: 3.261 - val acc: 6.667 [*]

Epoch: 30/200 - LR: 0.001000
train loss: 3.258 - train acc: 7.742 - val loss: 3.255 - val acc: 6.667 [*]

Epoch: 31/200 - LR: 0.001000
train loss: 3.249 - train acc: 7.742 - val loss: 3.257 - val acc: 6.667 [*]

Epoch: 32/200 - LR: 0.001000
train loss: 3.255 - train acc: 7.742 - val loss: 3.256 - val acc: 6.667 [*]

Epoch: 33/200 - LR: 0.001000
train loss: 3.249 - train acc: 7.742 - val loss: 3.256 - val acc: 6.667 [*]

Epoch: 34/200 - LR: 0.001000
train loss: 3.258 - train acc: 7.742 - val loss: 3.251 - val acc: 6.667 [*]

Epoch: 35/200 - LR: 0.001000
train loss: 3.242 - train acc: 7.742 - val loss: 3.249 - val acc: 6.667 [*]

Epoch: 36/200 - LR: 0.001000
train loss: 3.241 - train acc: 7.742 - val loss: 3.248 - val acc: 6.667 [*]

Epoch: 37/200 - LR: 0.001000
train loss: 3.245 - train acc: 7.742 - val loss: 3.244 - val acc: 6.667 [*]

Epoch: 38/200 - LR: 0.001000
train loss: 3.238 - train acc: 7.742 - val loss: 3.247 - val acc: 6.667 [*]

Epoch: 39/200 - LR: 0.001000
train loss: 3.241 - train acc: 7.742 - val loss: 3.243 - val acc: 6.667 [*]

Epoch: 40/200 - LR: 0.001000
train loss: 3.232 - train acc: 7.742 - val loss: 3.243 - val acc: 6.667 [*]

Epoch: 41/200 - LR: 0.001000
train loss: 3.233 - train acc: 7.742 - val loss: 3.241 - val acc: 6.667 [*]

Epoch: 42/200 - LR: 0.001000
train loss: 3.230 - train acc: 7.834 - val loss: 3.239 - val acc: 6.667 [*]

Epoch: 43/200 - LR: 0.001000
train loss: 3.228 - train acc: 7.742 - val loss: 3.235 - val acc: 6.667 [*]

Epoch: 44/200 - LR: 0.001000
train loss: 3.229 - train acc: 7.834 - val loss: 3.235 - val acc: 6.667 [*]

Epoch: 45/200 - LR: 0.001000
train loss: 3.228 - train acc: 7.834 - val loss: 3.235 - val acc: 6.667 [*]

Epoch: 46/200 - LR: 0.001000
train loss: 3.228 - train acc: 7.742 - val loss: 3.236 - val acc: 6.667 [*]

Epoch: 47/200 - LR: 0.001000
train loss: 3.219 - train acc: 7.834 - val loss: 3.230 - val acc: 6.667 [*]

Epoch: 48/200 - LR: 0.001000
train loss: 3.225 - train acc: 7.742 - val loss: 3.230 - val acc: 6.667 [*]

Epoch: 49/200 - LR: 0.001000
train loss: 3.221 - train acc: 7.742 - val loss: 3.228 - val acc: 6.667 [*]

Epoch: 50/200 - LR: 0.001000
train loss: 3.219 - train acc: 8.018 - val loss: 3.228 - val acc: 6.667 [*]

Epoch: 51/200 - LR: 0.001000
train loss: 3.215 - train acc: 7.834 - val loss: 3.226 - val acc: 6.667 [*]

Epoch: 52/200 - LR: 0.001000
train loss: 3.215 - train acc: 7.926 - val loss: 3.227 - val acc: 6.667 [*]

Epoch: 53/200 - LR: 0.001000
train loss: 3.214 - train acc: 7.926 - val loss: 3.226 - val acc: 6.667 [*]

Epoch: 54/200 - LR: 0.001000
train loss: 3.210 - train acc: 8.295 - val loss: 3.224 - val acc: 6.667 [*]

Epoch: 55/200 - LR: 0.001000
train loss: 3.212 - train acc: 8.018 - val loss: 3.223 - val acc: 7.097 [*]

Epoch: 56/200 - LR: 0.001000
train loss: 3.211 - train acc: 8.018 - val loss: 3.220 - val acc: 6.882 [*]

Epoch: 57/200 - LR: 0.001000
train loss: 3.209 - train acc: 8.018 - val loss: 3.226 - val acc: 7.097 [*]

Epoch: 58/200 - LR: 0.001000
train loss: 3.205 - train acc: 8.203 - val loss: 3.226 - val acc: 7.097 [*]

Epoch: 59/200 - LR: 0.001000
train loss: 3.213 - train acc: 8.111 - val loss: 3.229 - val acc: 7.097 [*]

Epoch: 60/200 - LR: 0.001000
train loss: 3.216 - train acc: 8.848 - val loss: 3.232 - val acc: 7.097 [*]

Epoch: 61/200 - LR: 0.001000
train loss: 3.221 - train acc: 8.756 - val loss: 3.235 - val acc: 7.742 [*]

Epoch: 62/200 - LR: 0.001000
train loss: 3.232 - train acc: 8.940 - val loss: 3.245 - val acc: 7.957 [*]

Epoch: 63/200 - LR: 0.001000
train loss: 3.232 - train acc: 8.940 - val loss: 3.247 - val acc: 7.527 [*]

Epoch: 64/200 - LR: 0.001000
train loss: 3.226 - train acc: 8.848 - val loss: 3.248 - val acc: 7.957 [*]

Epoch: 65/200 - LR: 0.001000
train loss: 3.237 - train acc: 9.677 - val loss: 3.250 - val acc: 8.387 [*]

Epoch: 66/200 - LR: 0.001000
train loss: 3.242 - train acc: 9.217 - val loss: 3.250 - val acc: 8.172 [*]

Epoch: 67/200 - LR: 0.001000
train loss: 3.234 - train acc: 9.954 - val loss: 3.258 - val acc: 8.602 [*]

Epoch: 68/200 - LR: 0.001000
train loss: 3.238 - train acc: 10.230 - val loss: 3.258 - val acc: 8.602 [*]

Epoch: 69/200 - LR: 0.001000
train loss: 3.243 - train acc: 9.770 - val loss: 3.259 - val acc: 8.172 [*]

Epoch: 70/200 - LR: 0.001000
train loss: 3.253 - train acc: 10.599 - val loss: 3.269 - val acc: 9.032 [*]

Epoch: 71/200 - LR: 0.001000
train loss: 3.250 - train acc: 10.507 - val loss: 3.266 - val acc: 9.462 [*]

Epoch: 72/200 - LR: 0.001000
train loss: 3.251 - train acc: 10.599 - val loss: 3.272 - val acc: 9.247 [*]

Epoch: 73/200 - LR: 0.001000
train loss: 3.251 - train acc: 10.691 - val loss: 3.268 - val acc: 9.677 [*]

Epoch: 74/200 - LR: 0.001000
train loss: 3.247 - train acc: 10.783 - val loss: 3.271 - val acc: 9.892 [*]

Epoch: 75/200 - LR: 0.001000
train loss: 3.251 - train acc: 10.783 - val loss: 3.274 - val acc: 9.892 [*]

Epoch: 76/200 - LR: 0.001000
train loss: 3.246 - train acc: 10.691 - val loss: 3.270 - val acc: 9.462 [*]

Epoch: 77/200 - LR: 0.001000
train loss: 3.249 - train acc: 10.968 - val loss: 3.267 - val acc: 10.108 [*]

Epoch: 78/200 - LR: 0.001000
train loss: 3.250 - train acc: 10.876 - val loss: 3.272 - val acc: 10.538 [*]

Epoch: 79/200 - LR: 0.001000
train loss: 3.251 - train acc: 11.152 - val loss: 3.275 - val acc: 10.323 [*]

Epoch: 80/200 - LR: 0.001000
train loss: 3.254 - train acc: 11.613 - val loss: 3.276 - val acc: 10.538 [*]

Epoch: 81/200 - LR: 0.001000
train loss: 3.250 - train acc: 11.152 - val loss: 3.275 - val acc: 11.398 [*]

Epoch: 82/200 - LR: 0.001000
train loss: 3.251 - train acc: 11.521 - val loss: 3.274 - val acc: 11.613 [*]

Epoch: 83/200 - LR: 0.001000
train loss: 3.253 - train acc: 11.797 - val loss: 3.272 - val acc: 11.183 [*]

Epoch: 84/200 - LR: 0.001000
train loss: 3.249 - train acc: 12.074 - val loss: 3.276 - val acc: 11.398 [*]

Epoch: 85/200 - LR: 0.001000
train loss: 3.242 - train acc: 11.521 - val loss: 3.270 - val acc: 11.613 [*]

Epoch: 86/200 - LR: 0.001000
train loss: 3.250 - train acc: 11.613 - val loss: 3.271 - val acc: 11.613 [*]

Epoch: 87/200 - LR: 0.001000
train loss: 3.245 - train acc: 11.705 - val loss: 3.270 - val acc: 11.613 [*]

Epoch: 88/200 - LR: 0.001000
train loss: 3.243 - train acc: 11.705 - val loss: 3.271 - val acc: 11.828 [*]

Epoch: 89/200 - LR: 0.001000
train loss: 3.246 - train acc: 11.705 - val loss: 3.269 - val acc: 11.613 [*]

Epoch: 90/200 - LR: 0.001000
train loss: 3.243 - train acc: 11.613 - val loss: 3.269 - val acc: 11.828 [*]

Epoch: 91/200 - LR: 0.001000
train loss: 3.242 - train acc: 11.705 - val loss: 3.267 - val acc: 11.398 [*]

Epoch: 92/200 - LR: 0.001000
train loss: 3.232 - train acc: 10.968 - val loss: 3.260 - val acc: 11.398 [*]

Epoch: 93/200 - LR: 0.001000
train loss: 3.238 - train acc: 11.152 - val loss: 3.256 - val acc: 10.968 [*]

Epoch: 94/200 - LR: 0.001000
train loss: 3.239 - train acc: 11.982 - val loss: 3.258 - val acc: 11.183 [*]

Epoch: 95/200 - LR: 0.001000
train loss: 3.233 - train acc: 11.982 - val loss: 3.257 - val acc: 11.183 [*]

Epoch: 96/200 - LR: 0.001000
train loss: 3.223 - train acc: 11.613 - val loss: 3.253 - val acc: 9.677 [*]

Epoch: 97/200 - LR: 0.001000
train loss: 3.224 - train acc: 11.705 - val loss: 3.258 - val acc: 10.323 [*]

Epoch: 98/200 - LR: 0.001000
train loss: 3.229 - train acc: 11.521 - val loss: 3.252 - val acc: 10.753 [*]

Epoch: 99/200 - LR: 0.001000
train loss: 3.227 - train acc: 11.429 - val loss: 3.248 - val acc: 10.753 [*]

Epoch: 100/200 - LR: 0.001000
train loss: 3.218 - train acc: 11.705 - val loss: 3.249 - val acc: 10.753 [*]

Epoch: 101/200 - LR: 0.001000
train loss: 3.222 - train acc: 12.074 - val loss: 3.251 - val acc: 10.538 [*]

Epoch: 102/200 - LR: 0.001000
train loss: 3.215 - train acc: 11.521 - val loss: 3.246 - val acc: 10.968 [*]

Epoch: 103/200 - LR: 0.001000
train loss: 3.217 - train acc: 10.968 - val loss: 3.239 - val acc: 11.613 [*]

Epoch: 104/200 - LR: 0.001000
train loss: 3.216 - train acc: 11.797 - val loss: 3.237 - val acc: 11.183 [*]

Epoch: 105/200 - LR: 0.001000
train loss: 3.205 - train acc: 11.429 - val loss: 3.238 - val acc: 10.108 [*]

Epoch: 106/200 - LR: 0.001000
train loss: 3.205 - train acc: 11.429 - val loss: 3.236 - val acc: 9.892 [*]

Epoch: 107/200 - LR: 0.001000
train loss: 3.204 - train acc: 11.797 - val loss: 3.236 - val acc: 10.753 [*]

Epoch: 108/200 - LR: 0.001000
train loss: 3.196 - train acc: 11.705 - val loss: 3.236 - val acc: 10.753 [*]

Epoch: 109/200 - LR: 0.001000
train loss: 3.212 - train acc: 11.613 - val loss: 3.230 - val acc: 10.538 [*]

Epoch: 110/200 - LR: 0.001000
train loss: 3.198 - train acc: 11.429 - val loss: 3.227 - val acc: 10.538 [*]

Epoch: 111/200 - LR: 0.001000
train loss: 3.195 - train acc: 12.350 - val loss: 3.232 - val acc: 10.323 [*]

Epoch: 112/200 - LR: 0.001000
train loss: 3.192 - train acc: 10.599 - val loss: 3.220 - val acc: 10.323 [*]

Epoch: 113/200 - LR: 0.001000
train loss: 3.198 - train acc: 12.166 - val loss: 3.221 - val acc: 10.108 [*]

Epoch: 114/200 - LR: 0.001000
train loss: 3.188 - train acc: 12.811 - val loss: 3.232 - val acc: 9.677 [*]

Epoch: 115/200 - LR: 0.001000
train loss: 3.190 - train acc: 11.889 - val loss: 3.225 - val acc: 10.108 [*]

Epoch: 116/200 - LR: 0.001000
train loss: 3.187 - train acc: 11.797 - val loss: 3.218 - val acc: 10.108 [*]

Epoch: 117/200 - LR: 0.001000
train loss: 3.179 - train acc: 10.968 - val loss: 3.210 - val acc: 9.462 [*]

Epoch: 118/200 - LR: 0.001000
train loss: 3.184 - train acc: 11.152 - val loss: 3.207 - val acc: 9.892 [*]

Epoch: 119/200 - LR: 0.001000
train loss: 3.184 - train acc: 12.350 - val loss: 3.211 - val acc: 9.247 [*]

Epoch: 120/200 - LR: 0.001000
train loss: 3.171 - train acc: 10.507 - val loss: 3.194 - val acc: 9.247 [*]

Epoch: 121/200 - LR: 0.001000
train loss: 3.172 - train acc: 12.074 - val loss: 3.198 - val acc: 10.108 [*]

Epoch: 122/200 - LR: 0.001000
train loss: 3.164 - train acc: 11.336 - val loss: 3.196 - val acc: 9.677 [*]

Epoch: 123/200 - LR: 0.001000
train loss: 3.170 - train acc: 11.152 - val loss: 3.188 - val acc: 9.892 [*]

Epoch: 124/200 - LR: 0.001000
train loss: 3.161 - train acc: 11.060 - val loss: 3.186 - val acc: 9.892 [*]

Epoch: 125/200 - LR: 0.001000
train loss: 3.159 - train acc: 11.705 - val loss: 3.183 - val acc: 9.892 [*]

Epoch: 126/200 - LR: 0.001000
train loss: 3.148 - train acc: 11.152 - val loss: 3.181 - val acc: 9.892 [*]

Epoch: 127/200 - LR: 0.001000
train loss: 3.150 - train acc: 11.060 - val loss: 3.176 - val acc: 10.108 [*]

Epoch: 128/200 - LR: 0.001000
train loss: 3.148 - train acc: 11.521 - val loss: 3.176 - val acc: 9.892 [*]

Epoch: 129/200 - LR: 0.001000
train loss: 3.146 - train acc: 11.060 - val loss: 3.175 - val acc: 9.892 [*]

Epoch: 130/200 - LR: 0.001000
train loss: 3.134 - train acc: 11.889 - val loss: 3.179 - val acc: 10.538 [*]

Epoch: 131/200 - LR: 0.001000
train loss: 3.133 - train acc: 11.521 - val loss: 3.176 - val acc: 10.108 [*]

Epoch: 132/200 - LR: 0.001000
train loss: 3.140 - train acc: 10.968 - val loss: 3.164 - val acc: 9.462 [*]

Epoch: 133/200 - LR: 0.001000
train loss: 3.140 - train acc: 11.060 - val loss: 3.165 - val acc: 9.892 [*]

Epoch: 134/200 - LR: 0.001000
train loss: 3.132 - train acc: 11.982 - val loss: 3.164 - val acc: 9.677 [*]

Epoch: 135/200 - LR: 0.001000
train loss: 3.138 - train acc: 12.166 - val loss: 3.165 - val acc: 9.462 [*]

Epoch: 136/200 - LR: 0.001000
train loss: 3.128 - train acc: 11.521 - val loss: 3.148 - val acc: 9.462 [*]

Epoch: 137/200 - LR: 0.001000
train loss: 3.117 - train acc: 12.074 - val loss: 3.154 - val acc: 10.323 [*]

Epoch: 138/200 - LR: 0.001000
train loss: 3.121 - train acc: 13.088 - val loss: 3.157 - val acc: 9.462 [*]

Epoch: 139/200 - LR: 0.001000
train loss: 3.120 - train acc: 12.350 - val loss: 3.147 - val acc: 10.753 [*]

Epoch: 140/200 - LR: 0.001000
train loss: 3.111 - train acc: 12.166 - val loss: 3.137 - val acc: 9.677 [*]

Epoch: 141/200 - LR: 0.001000
train loss: 3.117 - train acc: 12.903 - val loss: 3.144 - val acc: 9.462 [*]

Epoch: 142/200 - LR: 0.001000
train loss: 3.116 - train acc: 11.705 - val loss: 3.139 - val acc: 9.892 [*]

Epoch: 143/200 - LR: 0.001000
train loss: 3.103 - train acc: 13.180 - val loss: 3.147 - val acc: 11.613 [*]

Epoch: 144/200 - LR: 0.001000
train loss: 3.110 - train acc: 12.074 - val loss: 3.137 - val acc: 10.323 [*]

Epoch: 145/200 - LR: 0.001000
train loss: 3.097 - train acc: 12.535 - val loss: 3.130 - val acc: 10.968 [*]

Epoch: 146/200 - LR: 0.001000
train loss: 3.092 - train acc: 12.350 - val loss: 3.129 - val acc: 10.968 [*]

Epoch: 147/200 - LR: 0.001000
train loss: 3.099 - train acc: 13.548 - val loss: 3.130 - val acc: 12.043 [*]

Epoch: 148/200 - LR: 0.001000
train loss: 3.083 - train acc: 13.088 - val loss: 3.130 - val acc: 11.398 [*]

Epoch: 149/200 - LR: 0.001000
train loss: 3.093 - train acc: 12.903 - val loss: 3.125 - val acc: 11.828 [*]

Epoch: 150/200 - LR: 0.001000
train loss: 3.077 - train acc: 12.811 - val loss: 3.117 - val acc: 11.398 [*]

Epoch: 151/200 - LR: 0.001000
train loss: 3.084 - train acc: 13.641 - val loss: 3.118 - val acc: 12.473 [*]

Epoch: 152/200 - LR: 0.001000
train loss: 3.075 - train acc: 14.194 - val loss: 3.123 - val acc: 12.688 [*]

Epoch: 153/200 - LR: 0.001000
train loss: 3.070 - train acc: 13.825 - val loss: 3.110 - val acc: 12.688 [*]

Epoch: 154/200 - LR: 0.001000
train loss: 3.064 - train acc: 13.733 - val loss: 3.108 - val acc: 12.688 [*]

Epoch: 155/200 - LR: 0.001000
train loss: 3.065 - train acc: 13.825 - val loss: 3.106 - val acc: 11.828 [*]

Epoch: 156/200 - LR: 0.001000
train loss: 3.065 - train acc: 14.378 - val loss: 3.107 - val acc: 12.903 [*]

Epoch: 157/200 - LR: 0.001000
train loss: 3.051 - train acc: 13.364 - val loss: 3.099 - val acc: 12.473 [*]

Epoch: 158/200 - LR: 0.001000
train loss: 3.046 - train acc: 14.378 - val loss: 3.100 - val acc: 12.903 [*]

Epoch: 159/200 - LR: 0.001000
train loss: 3.059 - train acc: 14.470 - val loss: 3.097 - val acc: 13.548 [*]

Epoch: 160/200 - LR: 0.001000
train loss: 3.039 - train acc: 13.364 - val loss: 3.092 - val acc: 12.903 [*]

Epoch: 161/200 - LR: 0.001000
train loss: 3.042 - train acc: 14.286 - val loss: 3.088 - val acc: 12.258 [*]

Epoch: 162/200 - LR: 0.001000
train loss: 3.031 - train acc: 13.548 - val loss: 3.092 - val acc: 12.903 [*]

Epoch: 163/200 - LR: 0.001000
train loss: 3.050 - train acc: 13.825 - val loss: 3.087 - val acc: 13.118 [*]

Epoch: 164/200 - LR: 0.001000
train loss: 3.027 - train acc: 14.562 - val loss: 3.080 - val acc: 12.688 [*]

Epoch: 165/200 - LR: 0.001000
train loss: 3.023 - train acc: 14.470 - val loss: 3.081 - val acc: 13.118 [*]

Epoch: 166/200 - LR: 0.001000
train loss: 3.027 - train acc: 14.009 - val loss: 3.075 - val acc: 12.688 [*]

Epoch: 167/200 - LR: 0.001000
train loss: 3.028 - train acc: 14.286 - val loss: 3.073 - val acc: 12.258 [*]

Epoch: 168/200 - LR: 0.001000
train loss: 3.034 - train acc: 14.286 - val loss: 3.063 - val acc: 12.473 [*]

Epoch: 169/200 - LR: 0.001000
train loss: 3.010 - train acc: 14.470 - val loss: 3.066 - val acc: 13.118 [*]

Epoch: 170/200 - LR: 0.001000
train loss: 3.016 - train acc: 14.194 - val loss: 3.068 - val acc: 13.333 [*]

Epoch: 171/200 - LR: 0.001000
train loss: 3.007 - train acc: 14.562 - val loss: 3.059 - val acc: 13.548 [*]

Epoch: 172/200 - LR: 0.001000
train loss: 3.006 - train acc: 13.641 - val loss: 3.062 - val acc: 12.903 [*]

Epoch: 173/200 - LR: 0.001000
train loss: 2.992 - train acc: 14.194 - val loss: 3.060 - val acc: 14.409 [*]

Epoch: 174/200 - LR: 0.001000
train loss: 3.001 - train acc: 13.272 - val loss: 3.053 - val acc: 13.978 [*]

Epoch: 175/200 - LR: 0.001000
train loss: 3.000 - train acc: 14.101 - val loss: 3.046 - val acc: 12.903 [*]

Epoch: 176/200 - LR: 0.001000
train loss: 2.998 - train acc: 14.009 - val loss: 3.043 - val acc: 13.978 [*]

Epoch: 177/200 - LR: 0.001000
train loss: 2.991 - train acc: 14.839 - val loss: 3.051 - val acc: 14.409 [*]

Epoch: 178/200 - LR: 0.001000
train loss: 2.987 - train acc: 14.009 - val loss: 3.046 - val acc: 13.978 [*]

Epoch: 179/200 - LR: 0.001000
train loss: 2.990 - train acc: 15.207 - val loss: 3.057 - val acc: 14.839 [*]

Epoch: 180/200 - LR: 0.001000
train loss: 3.003 - train acc: 14.378 - val loss: 3.052 - val acc: 13.333 [*]

Epoch: 181/200 - LR: 0.001000
train loss: 2.987 - train acc: 14.194 - val loss: 3.048 - val acc: 13.333 [*]

Epoch: 182/200 - LR: 0.001000
train loss: 2.971 - train acc: 14.562 - val loss: 3.038 - val acc: 14.409 [*]

Epoch: 183/200 - LR: 0.001000
train loss: 2.976 - train acc: 13.917 - val loss: 3.037 - val acc: 14.194 [*]

Epoch: 184/200 - LR: 0.001000
train loss: 2.985 - train acc: 14.101 - val loss: 3.030 - val acc: 13.763 [*]

Epoch: 185/200 - LR: 0.001000
train loss: 2.979 - train acc: 14.286 - val loss: 3.031 - val acc: 13.763 [*]

Epoch: 186/200 - LR: 0.001000
train loss: 2.965 - train acc: 14.562 - val loss: 3.025 - val acc: 14.194 [*]

Epoch: 187/200 - LR: 0.001000
train loss: 2.965 - train acc: 15.392 - val loss: 3.038 - val acc: 13.978 [*]

Epoch: 188/200 - LR: 0.001000
train loss: 2.977 - train acc: 14.747 - val loss: 3.030 - val acc: 14.409 [*]

Epoch: 189/200 - LR: 0.001000
train loss: 2.957 - train acc: 13.456 - val loss: 3.022 - val acc: 13.763 [*]

Epoch: 190/200 - LR: 0.001000
train loss: 2.948 - train acc: 13.917 - val loss: 3.021 - val acc: 14.194 [*]

Epoch: 191/200 - LR: 0.001000
train loss: 2.950 - train acc: 13.548 - val loss: 3.019 - val acc: 13.333 [*]

Epoch: 192/200 - LR: 0.001000
train loss: 2.944 - train acc: 14.378 - val loss: 3.022 - val acc: 15.054 [*]

Epoch: 193/200 - LR: 0.001000
train loss: 2.953 - train acc: 13.364 - val loss: 3.018 - val acc: 15.699 [*]

Epoch: 194/200 - LR: 0.001000
train loss: 2.945 - train acc: 13.917 - val loss: 3.018 - val acc: 13.763 [*]

Epoch: 195/200 - LR: 0.001000
train loss: 2.949 - train acc: 14.101 - val loss: 3.014 - val acc: 13.763 [*]

Epoch: 196/200 - LR: 0.001000
train loss: 2.950 - train acc: 14.654 - val loss: 3.017 - val acc: 14.409 [*]

Epoch: 197/200 - LR: 0.001000
train loss: 2.954 - train acc: 15.115 - val loss: 3.022 - val acc: 13.548 [*]

Epoch: 198/200 - LR: 0.001000
train loss: 2.951 - train acc: 14.654 - val loss: 3.022 - val acc: 14.194 [*]

Epoch: 199/200 - LR: 0.001000
train loss: 2.949 - train acc: 15.115 - val loss: 3.036 - val acc: 14.194 [*]

Epoch: 200/200 - LR: 0.001000
train loss: 2.946 - train acc: 14.470 - val loss: 3.022 - val acc: 13.978 [*]
[*] Test Acc: 97/647 (14.00%)

Process finished with exit code 0
'''