import os
import torch
from pathlib import Path
from torchvision.datasets import VisionDataset
from .load_img import load_img
from .condition_map import condition_map


# Define the module exports
__all__ = ['SynSELMA']


# CONSTANTS
CATEGORY_MAP = {"selma": "car", "syndrone": "drone"}


class SynSELMA(VisionDataset):
    # Defien the category of the data inside the dataset
    category = "mixed"

    # Define if the dataset is weather aware or not
    weather_aware = True

    # Defien the directories for the source dataset
    dataset_dir = {"syndrone": Path("syndrone/renders"), "selma": Path("SELMA_LADD")}
    syndrone_split_dir = (
        Path(__file__).parent.parent.parent / "data/syndrone/splits"
    )
    selma_split_dir = Path(__file__).parent.parent.parent / "data/selma/splits"

    # Define the type of the dataset
    ds_type = "supervised"

    def __init__(
        self,
        root,
        transform=None,
        test_transform=None,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        split="train",
    ):
        super().__init__(root, transform=transform)

        self.root = root
        self.test = split not in ('train',)
        self.transform = transform
        self.test_transform = test_transform
        self.mean = mean
        self.std = std
        self.paths = []
        self.return_unprocessed_image = False

        # Define the possible weather conditions
        self.weather_conditions = ["clear", "night", "rain", "fog"]

        # Define the var that will contain the dict that links each index to the correct weather
        self.classes = None

        # Load the correct split
        # Syndorne
        syndrone_splits = {
            val: self.syndrone_split_dir / f"{split}_{val}.txt"
            for val in self.weather_conditions
        }
        # SELMA
        selma_splits = {
            val: self.selma_split_dir / f"{split}_{val}.txt"
            for val in self.weather_conditions
        }
        splits = {"syndrone": syndrone_splits, "selma": selma_splits}

        # Initialize the classes dict
        classes = {
            "selma": {val: [] for val in self.weather_conditions},
            "syndrone": {val: [] for val in self.weather_conditions},
        }

        # Load the content of the splits and fill the classes dict
        data = []
        last_len = 0
        for dts in ["syndrone", "selma"]:
            for cond in self.weather_conditions:
                with open(splits[dts][cond]) as f:
                    lines = f.read().splitlines()
                    classes[dts][cond] += range(last_len, len(lines) + last_len)
                    last_len += len(lines)
                    data += [
                        {
                            f"{dts}_{cond}": f'{self.dataset_dir[dts]}/{l.split(" ")[0]} {self.dataset_dir[dts]}/{l.split(" ")[1]}'
                        }
                        for l in lines
                    ]

        # Save the data in the right variable
        self.paths = data
        self.classes = classes

    def __getitem__(self, index):
        transform = self.transform if not self.test else self.test_transform
        condition = list(self.paths[index].keys())[0].split('_')
        condition = f"{CATEGORY_MAP[condition[0]]}_{condition[1]}"
        x_path, y_path = list(self.paths[index].values())[0].split(" ")
        x, y = load_img(
            x_path=os.path.join(self.root, x_path.lstrip("/")),
            y_path=os.path.join(self.root, y_path.lstrip("/")),
        )
        condition = condition_map(condition)

        if self.return_unprocessed_image:
            return x, condition
        
        x, y = transform(x, y)
        condition = torch.tensor(condition)

        return x, condition
    
    def __len__(self):
        return len(self.paths)
    
    def set_style_tf_fn(self, style_tf_fn):
        self.style_tf_fn = style_tf_fn

    def reset_style_tf_fn(self):
        self.style_tf_fn = None

    def get_classes(self):
        return self.classes
    
    def get_category(self):
        return self.category
