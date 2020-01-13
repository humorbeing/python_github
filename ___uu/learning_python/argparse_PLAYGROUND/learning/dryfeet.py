import math
import argparse

def fib(n):
    a, b = 0, 1
    for i in range(n):
        a, b = b, a+b
    return a


def Main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_test',
        help="The fib number you wish to calculate.",
        type=int
    )
    parser.add_argument(
        '-o',
        '--output',
        help='Output result to a file.',
        action='store_true'
    )
    args = parser.parse_args()

    # result = fib(args.num)
    # print("The "+str(args.num+"th fib number is "+ str(result))
    result=args.num_test
    if args.output:
        print(result)


if __name__ == '__main__':
    Main()