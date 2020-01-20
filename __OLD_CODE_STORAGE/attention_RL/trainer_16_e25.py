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
        self.patch_size = 16
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
        self.epochs = 25
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


'''/media/ray/SSD/workspace/python/ENVIRONMENT/python3.5/bin/python3.5 /media/ray/SSD/workspace/python/attention_RL/recurrent-visual-attention/trainer_16_e25.py
These is no random seed given.
[*] Number of model parameters: 500,766

[*] Train on 1085 samples, validate on 465 samples

Epoch: 1/25 - LR: 0.001000
train loss: 3.311 - train acc: 5.253 - val loss: 3.299 - val acc: 5.376 [*]

Epoch: 2/25 - LR: 0.001000
train loss: 3.301 - train acc: 5.438 - val loss: 3.298 - val acc: 5.161 [*]

Epoch: 3/25 - LR: 0.001000
train loss: 3.302 - train acc: 5.438 - val loss: 3.296 - val acc: 5.161 [*]

Epoch: 4/25 - LR: 0.001000
train loss: 3.298 - train acc: 4.700 - val loss: 3.299 - val acc: 6.452 [*]

Epoch: 5/25 - LR: 0.001000
train loss: 3.300 - train acc: 5.161 - val loss: 3.296 - val acc: 7.097 [*]

Epoch: 6/25 - LR: 0.001000
train loss: 3.303 - train acc: 6.728 - val loss: 3.296 - val acc: 6.882 [*]

Epoch: 7/25 - LR: 0.001000
train loss: 3.309 - train acc: 6.820 - val loss: 3.296 - val acc: 5.806 [*]

Epoch: 8/25 - LR: 0.001000
train loss: 3.307 - train acc: 6.083 - val loss: 3.303 - val acc: 7.957 [*]

Epoch: 9/25 - LR: 0.001000
train loss: 3.319 - train acc: 7.097 - val loss: 3.311 - val acc: 7.527 [*]

Epoch: 10/25 - LR: 0.001000
train loss: 3.311 - train acc: 6.820 - val loss: 3.318 - val acc: 7.742 [*]

Epoch: 11/25 - LR: 0.001000
train loss: 3.326 - train acc: 7.465 - val loss: 3.321 - val acc: 6.667 [*]

Epoch: 12/25 - LR: 0.001000
train loss: 3.328 - train acc: 7.650 - val loss: 3.327 - val acc: 7.312 [*]

Epoch: 13/25 - LR: 0.001000
train loss: 3.331 - train acc: 7.097 - val loss: 3.326 - val acc: 8.602 [*]

Epoch: 14/25 - LR: 0.001000
train loss: 3.332 - train acc: 6.452 - val loss: 3.324 - val acc: 7.527 [*]

Epoch: 15/25 - LR: 0.001000
train loss: 3.322 - train acc: 6.912 - val loss: 3.317 - val acc: 7.527 [*]

Epoch: 16/25 - LR: 0.001000
train loss: 3.324 - train acc: 7.005 - val loss: 3.322 - val acc: 7.527 [*]

Epoch: 17/25 - LR: 0.001000
train loss: 3.323 - train acc: 7.189 - val loss: 3.318 - val acc: 8.387 [*]

Epoch: 18/25 - LR: 0.001000
train loss: 3.317 - train acc: 6.728 - val loss: 3.316 - val acc: 7.742 [*]

Epoch: 19/25 - LR: 0.001000
train loss: 3.312 - train acc: 7.558 - val loss: 3.315 - val acc: 7.097 [*]

Epoch: 20/25 - LR: 0.001000
train loss: 3.312 - train acc: 6.820 - val loss: 3.315 - val acc: 7.742 [*]

Epoch: 21/25 - LR: 0.001000
train loss: 3.303 - train acc: 5.622 - val loss: 3.305 - val acc: 7.957 [*]

Epoch: 22/25 - LR: 0.001000
train loss: 3.304 - train acc: 7.465 - val loss: 3.305 - val acc: 6.882 [*]

Epoch: 23/25 - LR: 0.001000
train loss: 3.300 - train acc: 7.834 - val loss: 3.305 - val acc: 7.097 [*]

Epoch: 24/25 - LR: 0.001000
train loss: 3.307 - train acc: 7.558 - val loss: 3.305 - val acc: 7.097 [*]

Epoch: 25/25 - LR: 0.001000
train loss: 3.297 - train acc: 7.373 - val loss: 3.301 - val acc: 7.527 [*]
[*] Test Acc: 50/647 (7.00%)

Process finished with exit code 0
'''