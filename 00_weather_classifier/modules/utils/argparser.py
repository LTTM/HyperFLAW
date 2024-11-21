import argparse
from pathlib import Path


# Define the expoted
__all__ = ['argparser']


# Constants
MODELS = ['conv', 'mobile', 'resnet']


# Auxiliary functions
def str2path(string):
    return Path(string).absolute()


def argparser(script: str):
    parser = argparse.ArgumentParser(description='Description of your program')

    if script == 'train':
        required_ckpt = False
        required_txt = False
    elif script == 'eval':
        required_ckpt = True
        required_txt = True
    else:
        raise ValueError(f'Invalid script name: {script}')

    parser.add_argument('--num-epochs', '-e', type=int, default=10, help='Number of epochs')
    parser.add_argument('--dataset-path', '-d', type=str2path, required=True, help='Path to the dataset')
    parser.add_argument('--checkpoint-path', '-C', type=str2path, required=required_ckpt, help='Path to the checkpoints')
    parser.add_argument('--dts-txt', '-t', type=str2path, required=required_txt, help='Path to the dataset txt file')
    parser.add_argument('--batch-size', '-b', type=int, default=16, help='Batch size')
    parser.add_argument("--lr", "-l", type=float, default=0.001, help="Learning rate")
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum')
    parser.add_argument('--model', '-M', type=str, required=True, choices=MODELS, help='Model to use')
    parser.add_argument('--color-jitter', '-c', action='store_true', default=True, help='Enable color jitter')
    parser.add_argument('--gaussian-blur', '-g', action='store_true', default=True, help='Enable gaussian blur')
    parser.add_argument('--random-flip', '-f', action='store_true', default=True, help='Enable random flip')

    return parser.parse_args()
