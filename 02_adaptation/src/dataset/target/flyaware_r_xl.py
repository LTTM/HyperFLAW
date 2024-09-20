import os
import numpy as np
from dataset.target.target_dataset import TargetDataset
from ..load_img import load_img


class FLYAWARERXL(TargetDataset):
    task = 'segmentation'
    ds_type = 'unsupervised'
    category = "drone"

    dataset_dir = "FLYAWARE-R-XL"

    labels2train = {0: 2, 1: 0, 2: 13, 3: 8, 4: 8, 5: 13, 6: 11, 7: 255}
    # 0 building -> building 2
    # 1 road     -> road     0
    # 2 s. car   -> car      13
    # 3 tree     -> veg.     8
    # 4 low veg. -> veg.     8
    # 5 m. car   -> car      13
    # 6 human    -> person   11
    # 7 bg       -> bg       255

    labels2coarse = {0:0, 1:0, 2:2, 3:18, 4:18, 5:18, 6:18, 7:18, 8:8, 9:8, 10:18, 11:11, 12:11, 13:13, 14:13, 15:13, 16:18, 17:18, 18:18}
    # bg == 18
    # 0: "road"
    # 1: "sidewalk" -> road
    # 2: "building"
    # 3: "wall" -> building
    # 4: "fence" -> building
    # 5: "pole" -> road
    # 6: "traffic light" -> bg
    # 7: "traffic sign" -> bg
    # 8: "vegetation"
    # 9: "terrain" -> vegetation
    # 10: "sky" -> bg
    # 11: "person"
    # 12: "rider" -> person
    # 13: "car"
    # 14: "truck" -> car
    # 15: "bus" -> car
    # 16: "train" -> car
    # 17: "motorbike" -> person
    # 18: "bicycle" -> person

    weather_map = {"day": "clear", "night": "night", "rain": "rain", "fog": "fog"}

    # Define if the dataset is weather aware or not
    weather_aware = True

    images_dir = os.path.join(dataset_dir, "rgb")
    target_dir = os.path.join(dataset_dir, "gt")

    def __init__(self, paths, root, transform=None, test_transform=None, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                 cv2=False, hp_filtered=False):
        super().__init__(paths, root, transform=transform, test_transform=test_transform, mean=mean, std=std, cv2=cv2,
                         hp_filtered=hp_filtered)
    
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

    def _get_images(self, index):
        x_path = os.path.join(self.root, self.images_dir, self.paths['x'][index])
        x_hpf_path = x_path.replace('data', 'hp_filtered') if self.hp_filtered else None

        if self.ds_type == 'supervised' or self.test:
            try:
                y_path = os.path.join(self.root, self.target_dir, self.paths['y'][index])
            except IndexError:
                y_path = None
        else:
            y_path = None
        x, y, x_hpf = load_img(x_path=x_path, y_path=y_path, cv2=self.cv2, x_hpf_path=x_hpf_path, encoding=self._encode_segmap)
        return x, y, x_hpf
