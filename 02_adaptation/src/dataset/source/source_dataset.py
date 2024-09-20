import os
import torch
import numpy as np

from torch import from_numpy
from abc import ABCMeta, abstractmethod
from torchvision.datasets import VisionDataset
from ..load_img import load_img


# Constants
CLASS_NAME = ["road",
              "sidewalk",
              "building",
              "wall",
              "fence",
              "pole",
              "traffic light",
              "traffic sign",
              "vegetation",
              "terrain",
              "sky",
              "person",
              "rider",
              "car",
              "truck",
              "bus",
              "train",
              "motorbike",
              "bicycle"]


class SourceDataset(VisionDataset, metaclass=ABCMeta):

    def __init__(self, root, transform=None, test_transform=None, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                 cv2=False, split='train'):

        super().__init__(root, transform=transform, target_transform=None)

        self.root = root
        self.test = split not in ('train',)
        self.transform = transform
        self.test_transform = test_transform
        self.mean = mean
        self.std = std
        self.cv2 = cv2

        self.target_transform = self.__map_labels()
        self.paths = []

        # Define the possible weather conditions
        self.weather_conditions = ["clear", "night", "rain", "fog"]

        # Define the var that will contain the dict that links each index to the correct weather
        self.classes = None

        self.return_unprocessed_image = False
        self.style_tf_fn = None

    def set_style_tf_fn(self, style_tf_fn):
        self.style_tf_fn = style_tf_fn

    def reset_style_tf_fn(self):
        self.style_tf_fn = None

    def __getitem__(self, index):
        transform = self.transform if not self.test else self.test_transform
        condition = None
        x_path, y_path = self.paths[index].strip('\n').split(' ')
        x, y, _ = load_img(
            x_path=os.path.join(self.root, self.dataset_dir, 'data', x_path.lstrip("/")),
            y_path=os.path.join(self.root, self.dataset_dir, 'data', y_path.lstrip("/")),
            cv2=self.cv2
        )
        if self.return_unprocessed_image:
            return x
        if self.style_tf_fn is not None:
            x = self.style_tf_fn(x)
        x, y = transform(x, y)
        y = self.target_transform(y)

        if self.test:
            plot_x = torch.clone(x)
            return (plot_x, x), y, condition

        return x, y, condition

    def __len__(self):
        return len(self.paths)

    def __map_labels(self, labels2train=None):
        if labels2train is None:
            labels2train = self.labels2train
        mapping = np.zeros((256,), dtype=np.int64) + 255
        for k, v in labels2train.items():
            mapping[k] = v
        return lambda x: from_numpy(mapping[x])

    def get_classes(self):
        return self.classes
    
    def get_category(self):
        return self.category

    @property
    @abstractmethod
    def labels2train(self):
        pass

    # Property that defien the category of the data inside the dataset
    @property
    @abstractmethod
    def category(self):
        pass

    # Property that define if the dataset is weather aware or not
    @property
    @abstractmethod
    def weather_aware(self):
        pass

    @property
    @abstractmethod
    def dataset_dir(self):
        pass
