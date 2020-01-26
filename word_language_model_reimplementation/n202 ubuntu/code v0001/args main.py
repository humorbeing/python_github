import argparse

from runner import main



def ss(s):
    print(s)
    import sys
    sys.exit(1)
# import os
# print(os.getcwd())
parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
# parser.add_argument('--data', type=str, default='./data/wikitext-2',
#                     help='location of the data corpus')
# parser.add_argument('--model', type=str, default='LSTM',
#                     help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
# parser.add_argument('--emsize', type=int, default=200,
#                     help='size of word embeddings')
# parser.add_argument('--nhid', type=int, default=200,
#                     help='number of hidden units per layer')
# parser.add_argument('--nlayers', type=int, default=2,
#                     help='number of layers')
# parser.add_argument('--lr', type=float, default=20,
#                     help='initial learning rate')
# parser.add_argument('--clip', type=float, default=0.25,
#                     help='gradient clipping')
# parser.add_argument('--epochs', type=int, default=40,
#                     help='upper epoch limit')
# parser.add_argument('--batch_size', type=int, default=20, metavar='N',
#                     help='batch size')
# parser.add_argument('--bptt', type=int, default=35,
#                     help='sequence length')
# parser.add_argument('--dropout', type=float, default=0.2,
#                     help='dropout applied to layers (0 = no dropout)')
# parser.add_argument('--tied', action='store_true',
#                     help='tie the word embedding and softmax weights')
# parser.add_argument('--seed', type=int, default=1111,
#                     help='random seed')
# parser.add_argument('--cuda', action='store_true',
#                     help='use CUDA')
# parser.add_argument('--log-interval', type=int, default=200, metavar='N',
#                     help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')

# parser.add_argument('--nhead', type=int, default=2,
#                     help='the number of heads in the encoder/decoder of the transformer model')

# args = parser.parse_args()
args = argparse.Namespace()
args.seed = 1111
args.is_cuda = True
args.data_root = '../../language-model-data'
args.batch_size = 20
args.model = 'LSTM'  # RNN_TANH, RNN_RELU, LSTM, GRU, Transformer
args.emsize = 200
args.nhid = 200
args.nlayers = 2
args.lr = 20
args.lr_adam = 0.001
args.dropout = 0.2
args.tied = False
args.epoch = 40
args.bptt = 35
args.clip = 0.25
args.log_interval = 200
args.nhead = 2
args.is_manual_update = True
args.wandb = None  # None means no. add project name
# wandb project: world-language-model-pytorch-examples
args.early_stop = 10
args.is_quickrun = False

# ===========================================
# args.wandb = 'world-language-model-pytorch-examples'
args.is_quickrun = True
args.epoch = 5
args.is_manual_update = False
args.model = 'LSTM'
main(args)
# args.model = 'GRU'
# main(args)
# args.model = 'RNN_TANH'
# main(args)
# args.model = 'RNN_RELU'
# main(args)
# args.model = 'Transformer'
# main(args)
#
# args.is_manual_update = False
# args.model = 'LSTM'
# main(args)
# args.model = 'GRU'
# main(args)
# args.model = 'RNN_TANH'
# main(args)
# args.model = 'RNN_RELU'
# main(args)
# args.model = 'Transformer'
# main(args)