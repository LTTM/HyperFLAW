import os
import torch
from pathlib import Path

from dataset.source.source_dataset import SourceDataset
from ..load_img import load_img


class FLYSELMA(SourceDataset):
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

    # Define the possible weather conditions
    weather_conditions = ["clear", "night", "rain", "fog"]

    # Defien the directories for the source dataset
    dataset_dir = {"flyawares": Path("FLYAWARE-S/renders"), "selma": Path("SELMA_LADD")}
    flyawares_split_dir = (
        Path(__file__).parent.parent.parent.parent / "data/flyaware-s/splits"
    )
    selma_split_dir = Path(__file__).parent.parent.parent.parent / "data/selma/splits"

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
                dts in FLYSELMA.labels2train
            ), f"Class mapping missing for {target_dataset}, choose from: {FLYSELMA.labels2train.keys()}"
        if len(target_dataset) > 1:
            self.labels2train = FLYSELMA.labels2train['cityscapes']
        else:
            self.labels2train = FLYSELMA.labels2train[target_dataset[0]]

        super().__init__(
            root,
            transform=transform,
            test_transform=test_transform,
            mean=mean,
            std=std,
            cv2=cv2,
            split=split,
        )

        if only_clear and split == "train":
            self.weather_conditions = ["clear"]

        # Load the correct split
        # Syndorne
        flyawares_splits = {
            val: self.flyawares_split_dir / f"{split}_{val}.txt"
            for val in self.weather_conditions
        }
        # SELMA
        selma_splits = {
            val: self.selma_split_dir / f"{split}_{val}.txt"
            for val in self.weather_conditions
        }
        splits = {"flyawares": flyawares_splits, "selma": selma_splits}

        # Load the content of the splits and fill the classes dict
        data = []
        for dts in ["flyawares", "selma"]:
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

    def __getitem__(self, index):
        transform = self.transform if not self.test else self.test_transform
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
        y = self.target_transform(y)

        if self.test:
            plot_x = torch.clone(x)
            return (plot_x, x), y

        return x, y
