from argparse import ArgumentParser
from argparse import Namespace

hi = Namespace()
print(hi)
hi.hi = 1
print(hi.hi)
def get_args():
    parser = ArgumentParser(description='PyTorch/torchtext SNLI example')
#    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    return args

args = get_args()
print(args)
#print(args.epochs)

args.hi = 10

print(args.hi)