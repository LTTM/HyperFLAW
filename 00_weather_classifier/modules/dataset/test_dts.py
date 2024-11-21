import torch
from torch.utils import data
from pathlib import Path
from .load_img import load_img
from .condition_map import condition_map
from ..utils.txt_loader import load_paths_from_txt


class TestDataset(data.Dataset):
    weather_conditions = ["clear", "night", "rain", "fog"]

    weather_map = {
        "clear_from_cityscapes": "clear",
        "day": "clear",
        "rain": "rain",
        "fog": "fog",
        "night": "night",
    }

    def __init__(
        self, txt_file, root, transform=None, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
    ):
        self.paths = load_paths_from_txt(txt_file)
        self.root = root
        self.transform = transform
        self.mean = mean
        self.std = std
        self.return_unprocessed_image = False

        if self.weather_map is not None:
            # Create a dictioanry containing the paths to the images splitted by its weather condition
            classes = {val: [] for val in self.weather_conditions}

            for i, img in enumerate(self.paths):
                cur_weather = self.weather_map[str(img).split("/")[2]]
                classes[cur_weather].append(i)

            self.classes = classes

    def __getitem__(self, index):
        x, _ = load_img(x_path=Path(self.root / self.paths[index]))

        if self.classes is not None:
            for key, value in self.classes.items():
                if index in value:
                    weather = key
                    break
        else:
            weather = "none"
        condition = condition_map(weather)
        condition = torch.tensor(condition)

        if self.return_unprocessed_image:
            return x

        out = self.transform(x)

        return out, condition

    def __len__(self):
        return len(self.paths)

    def get_category(self):
        return self.category

    def get_classes(self):
        return self.classes
