import math
import argparse

parser = argparse.ArgumentParser(
    description='Calculate volume of a Cylinder'
)
parser.add_argument(
    '-ra',
    '--radius-test',
    type=int,
    metavar='',
    help='Radius of Cylinder',
    required=True
)
parser.add_argument(
    '-he',
    '--height',
    type=int,
    help='Height of Cylinder',
    required=True
)
group = parser.add_mutually_exclusive_group()
group.add_argument(
    '-q',
    '--quiet',
    action='store_true',
    help='print quiet'
)
group.add_argument(
    '-v',
    '--verbose',
    action='store_true',
    help='print verbose'
)
args = parser.parse_args()


def cylinder_volume(radius, height):
    vol = (math.pi) * (radius ** 2) * (height)
    return vol


if __name__ == '__main__':
    vol = cylinder_volume(args.radius, args.height)
    if args.quiet:
        print(vol)
    elif args.verbose:
        print('Volume is', vol)
    else:
        print('looks like its', vol)
