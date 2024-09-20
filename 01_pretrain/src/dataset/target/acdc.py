import os
from .target_dataset import TargetDataset


class ACDC(TargetDataset):

    task = 'segmentation'
    ds_type = 'supervised'
    
    labels2train = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13,
                    27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

    images_dir = os.path.join('HyperFLAW-ACDC')
    target_dir = os.path.join('HyperFLAW-ACDC')

    def __init__(self, paths, root, transform=None, test_transform=None, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                 cv2=False, hp_filtered=False):
        super().__init__(paths, root, transform=transform, test_transform=test_transform, mean=mean, std=std, cv2=cv2,
                         hp_filtered=hp_filtered)
