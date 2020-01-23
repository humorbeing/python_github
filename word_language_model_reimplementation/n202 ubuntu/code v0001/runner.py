# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
import torch.optim as optim
import data
import model as Model
# import os
# print(os.getcwd())

def ss(s):
    print(s)
    import sys
    sys.exit(1)
def main(args):
    # print(os.getcwd())
    # ss('s')
    if args.wandb:  # using None as trigger. if not None, it should be project name
        import wandb
        wandb.init(project=args.wandb, reinit=True)
        wandb.config.update(args)

    torch.manual_seed(args.seed)
    # if torch.cuda.is_available():
    #     if not args.cuda:
    #         print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    device = torch.device("cuda" if args.is_cuda else "cpu")

    ###############################################################################
    # Load data
    ###############################################################################

    corpus = data.Corpus(args.data_root)

    # Starting from sequential data, batchify arranges the dataset into columns.
    # For instance, with the alphabet as the sequence and batch size 4, we'd get
    # ┌ a g m s ┐
    # │ b h n t │
    # │ c i o u │
    # │ d j p v │
    # │ e k q w │
    # └ f l r x ┘.
    # These columns are treated as independent by the model, which means that the
    # dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
    # batch processing.

    def batchify(data, bsz):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data.to(device)

    eval_batch_size = 10
    train_data = batchify(corpus.train, args.batch_size)
    val_data = batchify(corpus.valid, eval_batch_size)
    test_data = batchify(corpus.test, eval_batch_size)


    #
    ###############################################################################
    # Build the model
    ###############################################################################

    ntokens = len(corpus.dictionary)
    if args.model == 'Transformer':
        model = Model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
    else:
        model = Model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)

    criterion = nn.CrossEntropyLoss()
    if args.wandb:
        wandb.watch(model)
    #
    ###############################################################################
    # Training code
    ###############################################################################
    optimizer = optim.Adam(model.parameters(), lr=args.lr_adam)
    lmbda = lambda epoch: 0.95
    scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)
    # scheduler.step()
    # scheduler.step()
    # args.is_manual_update = True
    lr = args.lr
    def get_lr():
        if args.is_manual_update:
            output = 'M{:02.5f}'.format(lr)
        else:
            for p in optimizer.param_groups:
                output = 'A{:02.5f}'.format(p['lr'])
        return output
    # print(get_lr())

    # for p in optimizer.param_groups:
    #     print(p['lr'])
    #     # break
    # ss('-in main')
    def repackage_hidden(h):
        """Wraps hidden states in new Tensors, to detach them from their history."""

        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(repackage_hidden(v) for v in h)


    # get_batch subdivides the source data into chunks of length args.bptt.
    # If source is equal to the example output of the batchify function, with
    # a bptt-limit of 2, we'd get the following two Variables for i = 0:
    # ┌ a g m s ┐ ┌ b h n t ┐
    # └ b h n t ┘ └ c i o u ┘
    # Note that despite the name of the function, the subdivison of data is not
    # done along the batch dimension (i.e. dimension 1), since that was handled
    # by the batchify function. The chunks are along dimension 0, corresponding
    # to the seq_len dimension in the LSTM.

    def get_batch(source, i):
        seq_len = min(args.bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].view(-1)
        return data, target


    def evaluate(data_source):
        # Turn on evaluation mode which disables dropout.
        model.eval()
        total_loss = 0.
        ntokens = len(corpus.dictionary)
        if args.model != 'Transformer':
            hidden = model.init_hidden(eval_batch_size)
        with torch.no_grad():
            for i in range(0, data_source.size(0) - 1, args.bptt):
                data, targets = get_batch(data_source, i)
                if args.model == 'Transformer':
                    output = model(data)
                else:
                    output, hidden = model(data, hidden)
                    hidden = repackage_hidden(hidden)
                output_flat = output.view(-1, ntokens)
                total_loss += len(data) * criterion(output_flat, targets).item()
        return total_loss / (len(data_source) - 1)


    def train():
        # Turn on training mode which enables dropout.
        model.train()
        total_loss = 0.
        log_loss = 0.
        start_time = time.time()
        ntokens = len(corpus.dictionary)
        if args.model != 'Transformer':
            hidden = model.init_hidden(args.batch_size)
        for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
            data, targets = get_batch(train_data, i)
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            model.zero_grad()
            if args.model == 'Transformer':
                output = model(data)
            else:
                hidden = repackage_hidden(hidden)
                output, hidden = model(data, hidden)
            loss = criterion(output.view(-1, ntokens), targets)
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            if args.is_manual_update:
                for p in model.parameters():
                    p.data.add_(-lr, p.grad.data)
            else:
                optimizer.step()

            total_loss += loss.item()
            log_loss += len(data) * loss.item()
            if batch % args.log_interval == 0 and batch > 0:
                cur_loss = total_loss / args.log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // args.bptt, get_lr(),
                    elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()

            if args.is_quickrun:
                break
        return log_loss / (train_data.size(0)-1)
            # break


    # def export_onnx(path, batch_size, seq_len):
    #     print('The model is also exported in ONNX format at {}'.
    #           format(os.path.realpath(args.onnx_export)))
    #     model.eval()
    #     dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    #     hidden = model.init_hidden(batch_size)
    #     torch.onnx.export(model, (dummy_input, hidden), path)


    # Loop over epochs.

    best_val_loss = None
    early_stop_count = 0
    early_stop_when = 10
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, args.epoch+1):
            epoch_start_time = time.time()
            log_loss = train()
            # ss('-in main')
            val_loss = evaluate(val_data)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss, math.exp(val_loss)))
            print('-' * 89)
            if args.wandb:
                wandb.log({
                    'train loss': log_loss,
                    'valid loss': val_loss
                })
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                # with open(args.save, 'wb') as f:
                #     torch.save(model, f)
                best_val_loss = val_loss
                early_stop_count = 0
            else:
                if args.is_manual_update:
                    # Anneal the learning rate if no improvement has been seen in the validation dataset.
                    lr /= 4.0
                else:
                    scheduler.step()
                early_stop_count += 1
            if args.early_stop != None:
                print('early stop monitor [{}/{}]'.format(early_stop_count, args.early_stop))
                if early_stop_count > args.early_stop:
                    print('trigger early stop')
                    break
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    # with open(args.save, 'rb') as f:
    #     model = torch.load(f)
    #     # after load the rnn params are not a continuous chunk of memory
    #     # this makes them a continuous chunk, and will speed up forward pass
    #     # Currently, only rnn model supports flatten_parameters function.
    #     if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
    #         model.rnn.flatten_parameters()

    # Run on test data.
    test_loss = evaluate(test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)
    if args.wandb:
        wandb.log({'test loss':test_loss})
        wandb.join()
    # if len(args.onnx_export) > 0:
    #     # Export the model in ONNX format.
    #     export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)
