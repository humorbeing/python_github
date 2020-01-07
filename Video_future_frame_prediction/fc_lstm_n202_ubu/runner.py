import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import os
from util_args_log import Log



def runner(args, path):
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.is_cuda:
        print('CUDA is set to DO NOT USE')
        device = torch.device("cpu")

    data_set = np.load(path + 'mnist_test_seq.npy')
    test_set = data_set[:, :1000]
    train_set = data_set[:, 1000:7000]
    valid_set = data_set[:, 7000:]
    del data_set


    def input_target_maker(batch, device):
        batch = batch / 255.
        if args.is_standardization:
            batch = (batch - 0.5) / 0.5
        input_x = batch[:10, :, :, :]
        pred_target = batch[10:, :, :, :]
        rec_target = np.flip(batch[:10, :, :, :], axis=0)
        rec_target = np.ascontiguousarray(rec_target)
        input_x = torch.Tensor(input_x).to(device)
        pred_target = torch.Tensor(pred_target).to(device)
        rec_target = torch.Tensor(rec_target).to(device)
        return input_x, rec_target, pred_target

    if args.model == 'ED_R_01':
        from models import lstm_v0001 as m
    elif args.model == 'b':
        from models.ED_lstmcell_v0001 import FC_LSTM as m
    elif args.model == 'c':
        from models.ED_lstmcell_v0001 import FC_LSTM as m
    elif args.model == 'd':
        from models.ED_lstmcell_v0001 import FC_LSTM as m
    else:
        raise Exception('wrong model')
    model = m(args).to(device)
    optimizer = optim.Adam(model.parameters())

    log = Log(args)

    best_loss = 99999
    for e in range(args.epoch):
        train_rec_loss = []
        rec_loss = []
        pred_loss = []
        idx = np.random.permutation(len(train_set[0]))  # i.i.d. sampling
        for i in range(len(train_set[0]) // args.batch_size):
            model.train()
            input_x, rec_target, pred_target =\
                input_target_maker(
                    train_set[:, idx[i:i + args.batch_size]], device)

            optimizer.zero_grad()
            rec = model(input_x)
            # loss_recon = F.binary_cross_entropy(rec, rec_target)
            if args.mode == 'recon':
                if args.loss_function == 'mse':
                    loss_recon = F.mse_loss(rec, rec_target)
                elif args.loss_function == 'bce':
                    loss_recon = F.binary_cross_entropy(rec, rec_target)
                else:
                    raise Exception('wrong loss function')
            elif args.mode == 'pred':
                if args.loss_function == 'mse':
                    loss_recon = F.mse_loss(rec, pred_target)
                elif args.loss_function == 'bce':
                    loss_recon = F.binary_cross_entropy(rec, pred_target)
                else:
                    raise Exception('wrong loss function')
            else:
                raise Exception('wrong mode')
            loss = loss_recon
            loss.backward()
            optimizer.step()
            train_rec_loss.append(loss.item())
        e_train_rec_loss = np.mean(train_rec_loss)

        for i in range(len(valid_set[0]) // args.batch_size):
            with torch.no_grad():
                model.eval()
                input_x, rec_target, pred_target =\
                    input_target_maker(
                        valid_set[:, i:i + args.batch_size], device)
                rec = model(input_x)
                if args.mode == 'recon':
                    if args.loss_function == 'mse':
                        loss_recon = F.mse_loss(rec, rec_target)
                    elif args.loss_function == 'bce':
                        loss_recon = F.binary_cross_entropy(rec, rec_target)
                    else:
                        raise Exception('wrong loss function')
                elif args.mode == 'pred':
                    if args.loss_function == 'mse':
                        loss_recon = F.mse_loss(rec, pred_target)
                    elif args.loss_function == 'bce':
                        loss_recon = F.binary_cross_entropy(rec, pred_target)
                    else:
                        raise Exception('wrong loss function')
                else:
                    raise Exception('wrong mode')
                loss = loss_recon
            rec_loss.append(loss_recon.item())

        rec_l = np.mean(rec_loss)

        log_string = 'Epoch: {}, train_loss: {}, eval_loss: {}'.format(e, e_train_rec_loss, rec_l)
        log.log(log_string)
        if args.is_save:
            total_loss = rec_l
            if total_loss < best_loss:
                best_loss = total_loss

                if not os.path.exists(args.save_path):
                    os.makedirs(args.save_path)
                torch.save(model, args.model_save_file)