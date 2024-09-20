import os
import torch
import numpy as np
from torch import from_numpy
from pathlib import Path

from dataset.source.source_dataset import SourceDataset
from ..load_img import load_img


class ACFLY(SourceDataset):
    labels2train = {
        "flyawarer": {
            0: 2,
            1: 0,
            2: 13,
            3: 8,
            4: 8,
            5: 13,
            6: 11,
            7: 255
        },
        "acdc": {
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
            33: 18
        },
    }

    #label2coarse = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 2, 6: 4, 255:255}
    labels2coarse = {0:0, 1:0, 2:2, 3:18, 4:18, 5:18, 6:18, 7:18, 8:8, 9:8, 10:18, 11:11, 12:11, 13:13, 14:13, 15:13, 16:18, 17:18, 18:18}

    # Define the possible weather conditions
    weather_conditions = ["clear", "night", "rain", "fog"]

    # Defien the directories for the source dataset
    data_dir = Path(__file__).parent.parent.parent.parent / "data"
    dataset_dir = {"flyawarer": Path("FLYAWARE-R"), "acdc": Path("HyperFLAW-ACDC")}
    flyawarer_split_dir = data_dir / "flyawarer/splits"
    acdc_split_dir = data_dir / "acdc/splits"

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
    ):
        for dts in target_dataset:
            assert (
                dts in ACFLY.labels2train
            ), f"Class mapping missing for {target_dataset}, choose from: {ACFLY.labels2train.keys()}"
        if len(target_dataset) > 1:
            self.labels2train = ACFLY.labels2train['flyawarer']
        else:
            self.labels2train = ACFLY.labels2train[target_dataset[0]]

        super().__init__(
            root,
            transform=transform,
            test_transform=test_transform,
            mean=mean,
            std=std,
            cv2=cv2,
            split=split,
        )

        self.labels2train = ACFLY.labels2train
        self.target_transform = self.__map_labels

        # Load the correct split
        # Syndorne
        flyawarer_splits = {
            val: self.flyawarer_split_dir / f"{split}_{val}.txt"
            for val in self.weather_conditions
        }
        # SELMA
        acdc_splits = {
            val: self.acdc_split_dir / f"{split}_{val}.txt"
            for val in self.weather_conditions
        }
        splits = {"flyawarer": flyawarer_splits, "acdc": acdc_splits}

        # Load the content of the splits and fill the classes dict
        data = []
        for dts in ["flyawarer", "acdc"]:
            for cond in self.weather_conditions:
                with open(splits[dts][cond]) as f:
                    lines = f.read().splitlines()
                    data += [
                        {
                            f"{dts}_{cond}": f'{self.dataset_dir[dts]}/{l.split(" ")[0]} {self.dataset_dir[dts]}/{l.split(" ")[1]}'
                        }
                        for l in lines
                    ]

        # Save the data in the right variable
        self.paths = data

    @staticmethod
    def _encode_segmap(segcolors):
        """RGB colors to class labels"""
        colors = np.array([
            [128, 0, 0],  # Building
            [128, 64, 128],  # Road
            [192, 0, 192],  # Static car
            [0, 128, 0],  # Tree
            [128, 128, 0],  # Low vegetation
            [64, 0, 128],  # Moving car
            [64, 64, 0],  # Human
            [0, 0, 0]  # Background / clutter
        ], dtype=np.uint8)
        label_map = colors.shape[0]*np.ones((segcolors.shape[0], segcolors.shape[1]), dtype=np.uint8)
        for i, color in enumerate(colors):
            label_map[np.all(segcolors == color, axis=2)] = i
        return label_map

    def __getitem__(self, index):
        transform = self.transform if not self.test else self.test_transform
        dts_name = list(self.paths[index].keys())[0].split("_")[0]
        x_path, y_path = list(self.paths[index].values())[0].split(" ")

        encoding = self._encode_segmap if dts_name == "flyawarer" else None

        x, y, _ = load_img(
            x_path=os.path.join(self.root, x_path.lstrip("/")),
            y_path=os.path.join(self.root, y_path.lstrip("/")),
            cv2=self.cv2,
            encoding=encoding,
        )
        if self.return_unprocessed_image:
            return x
        if self.style_tf_fn is not None:
            x = self.style_tf_fn(x)
        x, y = transform(x, y)
        y = self.target_transform(y, dts_name=dts_name)

        if self.test:
            plot_x = torch.clone(x)
            return (plot_x, x), y

        return x, y, dts_name

    def __map_labels(self, x, dts_name):
        mapping = np.zeros((256,), dtype=np.int64) + 255
        for k, v in self.labels2train[dts_name].items():
            mapping[k] = v
        return from_numpy(mapping[x])