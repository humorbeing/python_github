from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='PyTorch/torchtext SNLI example')
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    return args

args = get_args()

print(args.epochs)

args.hi = 10

print(args.hi)