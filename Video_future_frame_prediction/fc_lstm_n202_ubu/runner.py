import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import os
from util_args_log import Log

def runner_sigmoid(args, path):
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.is_cuda:
        print('CUDA is set to DO NOT USE')
        device = torch.device("cpu")

    data_set = np.load(path + 'mnist_test_seq.npy')
    # test_set = data_set[:, :1000]
    # train_set = data_set[:, 1000:7000]
    # valid_set = data_set[:, 7000:]
    # test_set = data_set[:, :1000]
    train_set = data_set[:, :9000]
    valid_set = data_set[:, 9000:]
    del data_set
    if args.is_quickrun:
        train_set = train_set[:, :15]
        valid_set = valid_set[:, :10]
        args.batch_size = 5
        args.epoch = 2


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
        from models import lstm_copy as m
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
        batch_mse_recon_loss = []
        batch_bce_recon_loss = []
        batch_mse_pred_loss = []
        batch_bce_pred_loss = []
        idx = np.random.permutation(len(train_set[0]))  # i.i.d. sampling
        for i in range(len(train_set[0]) // args.batch_size):
            model.train()
            input_x, rec_target, pred_target =\
                input_target_maker(
                    train_set[:, idx[i:i + args.batch_size]], device)

            optimizer.zero_grad()
            rec1, rec2 = model(input_x)
            if args.mode == 'pred':
                mse_train_loss = F.mse_loss(rec1, pred_target)
                bce_train_loss = F.binary_cross_entropy(rec1, pred_target)
                batch_mse_recon_loss.append(0)
                batch_bce_recon_loss.append(0)
                batch_mse_pred_loss.append(mse_train_loss.item())
                batch_bce_pred_loss.append(bce_train_loss.item())
            elif args.mode == 'recon':
                mse_train_loss = F.mse_loss(rec1, rec_target)
                bce_train_loss = F.binary_cross_entropy(rec1, rec_target)
                batch_mse_recon_loss.append(mse_train_loss.item())
                batch_bce_recon_loss.append(bce_train_loss.item())
                batch_mse_pred_loss.append(0)
                batch_bce_pred_loss.append(0)
            else:  # 'both'
                mse_train_recon_loss = F.mse_loss(rec1, rec_target)
                bce_train_recon_loss = F.binary_cross_entropy(rec1, rec_target)
                mse_train_pred_loss = F.mse_loss(rec2, pred_target)
                bce_train_pred_loss = F.binary_cross_entropy(rec2, pred_target)
                batch_mse_recon_loss.append(mse_train_recon_loss.item())
                batch_bce_recon_loss.append(bce_train_recon_loss.item())
                batch_mse_pred_loss.append(mse_train_pred_loss.item())
                batch_bce_pred_loss.append(bce_train_pred_loss.item())
                mse_train_loss = args.recon_loss_lambda * mse_train_recon_loss + mse_train_pred_loss
                bce_train_loss = args.recon_loss_lambda * bce_train_recon_loss + bce_train_pred_loss

            if args.loss_function == 'mse':
                loss = mse_train_loss
            else:  # 'bce'
                loss = bce_train_loss
            loss.backward()
            optimizer.step()

        episode_train_mse_recon_loss = np.mean(batch_mse_recon_loss)
        episode_train_bce_recon_loss = np.mean(batch_bce_recon_loss)
        episode_train_mse_pred_loss = np.mean(batch_mse_pred_loss)
        episode_train_bce_pred_loss = np.mean(batch_bce_pred_loss)
        batch_mse_recon_loss = []
        batch_bce_recon_loss = []
        batch_mse_pred_loss = []
        batch_bce_pred_loss = []
        for i in range(len(valid_set[0]) // args.batch_size):
            with torch.no_grad():
                model.eval()
                input_x, rec_target, pred_target =\
                    input_target_maker(
                        valid_set[:, i:i + args.batch_size], device)
                rec1, rec2 = model(input_x)
                if args.mode == 'pred':
                    mse_train_loss = F.mse_loss(rec1, pred_target)
                    bce_train_loss = F.binary_cross_entropy(rec1, pred_target)
                    batch_mse_recon_loss.append(0)
                    batch_bce_recon_loss.append(0)
                    batch_mse_pred_loss.append(mse_train_loss.item())
                    batch_bce_pred_loss.append(bce_train_loss.item())
                elif args.mode == 'recon':
                    mse_train_loss = F.mse_loss(rec1, rec_target)
                    bce_train_loss = F.binary_cross_entropy(rec1, rec_target)
                    batch_mse_recon_loss.append(mse_train_loss.item())
                    batch_bce_recon_loss.append(bce_train_loss.item())
                    batch_mse_pred_loss.append(0)
                    batch_bce_pred_loss.append(0)
                else:  # 'both'
                    mse_train_recon_loss = F.mse_loss(rec1, rec_target)
                    bce_train_recon_loss = F.binary_cross_entropy(rec1, rec_target)
                    mse_train_pred_loss = F.mse_loss(rec2, pred_target)
                    bce_train_pred_loss = F.binary_cross_entropy(rec2, pred_target)
                    batch_mse_recon_loss.append(mse_train_recon_loss.item())
                    batch_bce_recon_loss.append(bce_train_recon_loss.item())
                    batch_mse_pred_loss.append(mse_train_pred_loss.item())
                    batch_bce_pred_loss.append(bce_train_pred_loss.item())

        episode_val_mse_recon_loss = np.mean(batch_mse_recon_loss)
        episode_val_bce_recon_loss = np.mean(batch_bce_recon_loss)
        episode_val_mse_pred_loss = np.mean(batch_mse_pred_loss)
        episode_val_bce_pred_loss = np.mean(batch_bce_pred_loss)
        # T: train, V: validation, M: mse, B: bce, R: recon, P: pred
        log_string = 'Epoch: {}, TMR: {}, TBR: {}, TMP: {}, TBP: {}, VMR: {}, VBR: {}, VMP: {}, VBP: {}'\
            .format(e, episode_train_mse_recon_loss, episode_train_bce_recon_loss,
                    episode_train_mse_pred_loss, episode_train_bce_pred_loss,
                    episode_val_mse_recon_loss, episode_val_bce_recon_loss,
                    episode_val_mse_pred_loss, episode_val_bce_pred_loss)
        log.log(log_string)
        if args.is_save:
            total_loss = episode_val_mse_recon_loss
            if total_loss < best_loss:
                best_loss = total_loss

                if not os.path.exists(args.save_path):
                    os.makedirs(args.save_path)
                torch.save(model, args.model_save_file)


def runner_no_bce_loss(args, path):
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.is_cuda:
        print('CUDA is set to DO NOT USE')
        device = torch.device("cpu")

    data_set = np.load(path + 'mnist_test_seq.npy')
    # test_set = data_set[:, :1000]
    # train_set = data_set[:, 1000:7000]
    # valid_set = data_set[:, 7000:]
    # test_set = data_set[:, :1000]
    train_set = data_set[:, :9000]
    valid_set = data_set[:, 9000:]
    del data_set
    if args.is_quickrun:
        train_set = train_set[:, :15]
        valid_set = valid_set[:, :10]
        args.batch_size = 5
        args.epoch = 2
    if args.loss_function == 'bce':
        # args.is_standardization = False
        # args.last_activation = 'sigmoid'  # 'tanh' / 'sigmoid' / 'non'
        args.loss_function = 'mse'

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
        from models import lstm_copy as m
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
        batch_mse_recon_loss = []
        batch_bce_recon_loss = []
        batch_mse_pred_loss = []
        batch_bce_pred_loss = []
        idx = np.random.permutation(len(train_set[0]))  # i.i.d. sampling
        for i in range(len(train_set[0]) // args.batch_size):
            model.train()
            input_x, rec_target, pred_target =\
                input_target_maker(
                    train_set[:, idx[i:i + args.batch_size]], device)

            optimizer.zero_grad()
            rec1, rec2 = model(input_x)
            if args.mode == 'pred':
                mse_train_loss = F.mse_loss(rec1, pred_target)
                # bce_train_loss = F.binary_cross_entropy(rec1, pred_target)
                batch_mse_recon_loss.append(0)
                batch_bce_recon_loss.append(0)
                batch_mse_pred_loss.append(mse_train_loss.item())
                batch_bce_pred_loss.append(0)
            elif args.mode == 'recon':
                mse_train_loss = F.mse_loss(rec1, rec_target)
                # bce_train_loss = F.binary_cross_entropy(rec1, rec_target)
                batch_mse_recon_loss.append(mse_train_loss.item())
                batch_bce_recon_loss.append(0)
                batch_mse_pred_loss.append(0)
                batch_bce_pred_loss.append(0)
            else:  # 'both'
                mse_train_recon_loss = F.mse_loss(rec1, rec_target)
                # bce_train_recon_loss = F.binary_cross_entropy(rec1, rec_target)
                mse_train_pred_loss = F.mse_loss(rec2, pred_target)
                # bce_train_pred_loss = F.binary_cross_entropy(rec2, pred_target)
                batch_mse_recon_loss.append(mse_train_recon_loss.item())
                batch_bce_recon_loss.append(0)
                batch_mse_pred_loss.append(mse_train_pred_loss.item())
                batch_bce_pred_loss.append(0)
                mse_train_loss = args.recon_loss_lambda * mse_train_recon_loss + mse_train_pred_loss
                # bce_train_loss = args.recon_loss_lambda * bce_train_recon_loss + bce_train_pred_loss


            loss = mse_train_loss

            loss.backward()
            optimizer.step()

        episode_train_mse_recon_loss = np.mean(batch_mse_recon_loss)
        episode_train_bce_recon_loss = np.mean(0)
        episode_train_mse_pred_loss = np.mean(batch_mse_pred_loss)
        episode_train_bce_pred_loss = np.mean(0)
        batch_mse_recon_loss = []
        batch_bce_recon_loss = []
        batch_mse_pred_loss = []
        batch_bce_pred_loss = []
        for i in range(len(valid_set[0]) // args.batch_size):
            with torch.no_grad():
                model.eval()
                input_x, rec_target, pred_target =\
                    input_target_maker(
                        valid_set[:, i:i + args.batch_size], device)
                rec1, rec2 = model(input_x)
                if args.mode == 'pred':
                    mse_train_loss = F.mse_loss(rec1, pred_target)
                    # bce_train_loss = F.binary_cross_entropy(rec1, pred_target)
                    batch_mse_recon_loss.append(0)
                    batch_bce_recon_loss.append(0)
                    batch_mse_pred_loss.append(mse_train_loss.item())
                    batch_bce_pred_loss.append(0)
                elif args.mode == 'recon':
                    mse_train_loss = F.mse_loss(rec1, rec_target)
                    # bce_train_loss = F.binary_cross_entropy(rec1, rec_target)
                    batch_mse_recon_loss.append(mse_train_loss.item())
                    batch_bce_recon_loss.append(0)
                    batch_mse_pred_loss.append(0)
                    batch_bce_pred_loss.append(0)
                else:  # 'both'
                    mse_train_recon_loss = F.mse_loss(rec1, rec_target)
                    # bce_train_recon_loss = F.binary_cross_entropy(rec1, rec_target)
                    mse_train_pred_loss = F.mse_loss(rec2, pred_target)
                    # bce_train_pred_loss = F.binary_cross_entropy(rec2, pred_target)
                    batch_mse_recon_loss.append(mse_train_recon_loss.item())
                    batch_bce_recon_loss.append(0)
                    batch_mse_pred_loss.append(mse_train_pred_loss.item())
                    batch_bce_pred_loss.append(0)

        episode_val_mse_recon_loss = np.mean(batch_mse_recon_loss)
        episode_val_bce_recon_loss = np.mean(0)
        episode_val_mse_pred_loss = np.mean(batch_mse_pred_loss)
        episode_val_bce_pred_loss = np.mean(0)
        # T: train, V: validation, M: mse, B: bce, R: recon, P: pred
        log_string = 'Epoch: {}, TMR: {}, TBR: {}, TMP: {}, TBP: {}, VMR: {}, VBR: {}, VMP: {}, VBP: {}'\
            .format(e, episode_train_mse_recon_loss, episode_train_bce_recon_loss,
                    episode_train_mse_pred_loss, episode_train_bce_pred_loss,
                    episode_val_mse_recon_loss, episode_val_bce_recon_loss,
                    episode_val_mse_pred_loss, episode_val_bce_pred_loss)
        log.log(log_string)
        if args.is_save:
            total_loss = episode_val_mse_recon_loss
            if total_loss < best_loss:
                best_loss = total_loss

                if not os.path.exists(args.save_path):
                    os.makedirs(args.save_path)
                torch.save(model, args.model_save_file)

def runner(arg, path):
    if arg.last_activation == 'sigmoid':
        arg.is_standardization = False
        runner_sigmoid(arg, path)
    else:
        arg.loss_function = 'mse'
        if arg.last_activation == 'tanh':
            arg.is_standardization = True
        runner_no_bce_loss(arg, path)