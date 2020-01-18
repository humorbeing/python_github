import argparse
main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
train_arg_parser.add_argument("--epochs", type=int, default=2,
                              help="number of training epochs, default is 2")
train_arg_parser.add_argument("--batch-size", type=int, default=4,
                              help="batch size for training, default is 4")


eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
eval_arg_parser.add_argument("--epochs", type=int, default=3,
                              help="number of training epochs, default is 2")

args = main_arg_parser.parse_args()
args.subcommand = 'eval'
print(args.subcommand)
print(args.epochs)
# raise Exception('hi')