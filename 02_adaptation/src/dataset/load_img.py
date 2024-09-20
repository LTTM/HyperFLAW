from PIL import Image
from cv2 import imread
import numpy as np


def load_img(x_path, y_path=None, cv2=False, x_hpf_path=None, encoding=None):
        if cv2:
            x = imread(x_path)[:, :, ::-1]
            y = imread(y_path, 0) if y_path is not None else None
            y = encoding(y) if encoding is not None else y
            x_hpf = imread(x_hpf_path) if x_hpf_path is not None else None
        else:
            x = Image.open(x_path)
            y = Image.open(y_path) if y_path is not None else None
            if encoding is not None and y is not None:
                y = np.array(y)
                y = encoding(y).astype(np.uint8)
                y = Image.fromarray(y)
            x_hpf = Image.open(x_hpf_path) if x_hpf_path is not None else None
        return x, y, x_hpf
