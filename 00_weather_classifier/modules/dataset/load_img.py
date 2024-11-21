from PIL import Image
import numpy as np


# Define the module exports
__all__ = ['load_img']


def load_img(x_path, y_path=None, encoding=None):
    x = Image.open(x_path)
    y = Image.open(y_path) if y_path is not None else None
    if encoding is not None and y is not None:
        y = np.array(y)
        y = encoding(y).astype(np.uint8)
        y = Image.fromarray(y)
    return x, y
