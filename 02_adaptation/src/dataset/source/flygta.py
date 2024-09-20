import os
import torch
from pathlib import Path
import numpy as np

from torch import from_numpy
from dataset.source.source_dataset import SourceDataset
from ..load_img import load_img


# CONSTANTS
CATEGORY_MAP = {"gta": "car", "flyawares": "drone"}


class FLYGTA(SourceDataset):
    labels2train = {
        "cityscapes": {
            4: 255,
            0: 255,
            1: 2,
            2: 4,
            3: 255,
            5: 5,
            6: 0,
            7: 0,
            8: 1,
            9: 8,
            10: 255,
            11: 3,
            12: 7,
            13: 10,
            14: 255,
            15: 255,
            16: 255,
            17: 255,
            18: 6,
            19: 255,
            20: 255,
            21: 255,
            22: 9,
            40: 11,
            41: 12,
            100: 13,
            101: 14,
            102: 15,
            103: 16,
            104: 17,
            105: 18,
        },
        "GTA2cityscapes": {
            7: 0,
            8: 1,
            11: 2,
            12: 3,
            13: 4,
            17: 5,
            19: 6,
            20: 7,
            21: 8,
            22: 9,
            23: 10,
            24: 11,
            25: 12,
            26: 13,
            27: 14,
            28: 15,
            31: 16,
            32: 17,
            33: 18,
        },
        "flyawarer": {
            1: 4,
            2: 4,
            3: 255,
            4: 0,
            5: 3,
            6: 4,
            7: 0,
            8: 0,
            9: 1,
            11: 4,
            12: 255,
            13: 255,
            14: 0,
            15: 4,
            16: 0,
            17: 255,
            18: 255,
            19: 255,
            20: 255,
            21: 255,
            22: 1,
            40: 2,
            41: 2,
            100: 3,
            101: 3,
            102: 3,
            103: 3,
            104: 3,
            105: 3,
        },
        "flyawarerxl": {
            1: 4,
            2: 4,
            3: 255,
            4: 0,
            5: 3,
            6: 4,
            7: 0,
            8: 0,
            9: 1,
            11: 4,
            12: 255,
            13: 255,
            14: 0,
            15: 4,
            16: 0,
            17: 255,
            18: 255,
            19: 255,
            20: 255,
            21: 255,
            22: 1,
            40: 2,
            41: 2,
            100: 3,
            101: 3,
            102: 3,
            103: 3,
            104: 3,
            105: 3,
        },
        "acdc": {
            4: 255,
            0: 255,
            1: 2,
            2: 4,
            3: 255,
            5: 5,
            6: 0,
            7: 0,
            8: 1,
            9: 8,
            10: 255,
            11: 3,
            12: 7,
            13: 10,
            14: 255,
            15: 255,
            16: 255,
            17: 255,
            18: 6,
            19: 255,
            20: 255,
            21: 255,
            22: 9,
            40: 11,
            41: 12,
            100: 13,
            101: 14,
            102: 15,
            103: 16,
            104: 17,
            105: 18,
        },
    }

    # Defien the category of the data inside the dataset
    category = "mixed"

    # Define if the dataset is weather aware or not
    weather_aware = False

    # Defien the directories for the source dataset
    dataset_dir = {"flyawares": Path("FLYAWARE-S/renders"), "gta": Path("gta5/data")}
    flyawares_split_dir = (
        Path(__file__).parent.parent.parent.parent / "data/flyaware-s/splits"
    )
    gta_split_dir = Path(__file__).parent.parent.parent.parent / "data/gta5/splits"

    # Define the type of the dataset
    ds_type = "supervised"

    def __init__(
        self,
        root,
        transform=None,
        test_transform=None,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        cv2=False,
        split="train",
        target_dataset=None,
        only_clear=False,
    ):
        for dts in target_dataset:
            assert (
                dts in FLYGTA.labels2train
            ), f"Class mapping missing for {target_dataset}, choose from: {FLYGTA.labels2train.keys()}"
        if len(target_dataset) > 1:
            self.labels2train = FLYGTA.labels2train["cityscapes"]
        else:
            self.labels2train = FLYGTA.labels2train[target_dataset[0]]

        super().__init__(
            root,
            transform=transform,
            test_transform=test_transform,
            mean=mean,
            std=std,
            cv2=cv2,
            split=split,
        )

        # Load the correct split
        splits = {
            "flyawares": self.flyawares_split_dir / f"{split}.txt",
            "gta": self.gta_split_dir / f"{split}.txt",
        }

        # Initialize the classes dict
        classes = {
            "gta": [],
            "flyawares": [],
        }

        # Load the content of the splits and fill the classes dict
        data = []
        last_len = 0
        for dts in ["flyawares", "gta"]:
            with open(splits[dts]) as f:
                lines = f.read().splitlines()
                classes[dts] += range(last_len, len(lines) + last_len)
                last_len += len(lines)
                data += [
                    {
                        f"{dts}": f'{self.dataset_dir[dts]}/{l.split(" ")[0]} {self.dataset_dir[dts]}/{l.split(" ")[1]}'
                    }
                    for l in lines
                ]

        # Save the data in the right variable
        self.paths = data
        self.classes = classes

    def __getitem__(self, index):
        transform = self.transform if not self.test else self.test_transform
        condition = list(self.paths[index].keys())[0]
        condition = f"{CATEGORY_MAP[condition]}_none"
        x_path, y_path = list(self.paths[index].values())[0].split(" ")
        x, y, _ = load_img(
            x_path=os.path.join(self.root, x_path.lstrip("/")),
            y_path=os.path.join(self.root, y_path.lstrip("/")),
            cv2=self.cv2,
        )
        if self.return_unprocessed_image:
            return x
        if self.style_tf_fn is not None:
            x = self.style_tf_fn(x)
        x, y = transform(x, y)
        y = self._map_labels(
            FLYGTA.labels2train[
                "GTA2cityscapes" if condition == "car_none" else "cityscapes"
            ],
        )(y)

        if self.test:
            plot_x = torch.clone(x)
            return (plot_x, x), y, condition

        return x, y, condition
    
    def _map_labels(self, labels2train=None):
        if labels2train is None:
            labels2train = self.labels2train
        mapping = np.zeros((256,), dtype=np.int64) + 255
        for k, v in labels2train.items():
            mapping[k] = v
        return lambda x: from_numpy(mapping[x])
