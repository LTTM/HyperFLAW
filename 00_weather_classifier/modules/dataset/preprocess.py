from functools import partial
from . import transform as tr
from .synselma import SynSELMA


# Define the module exports
__all__ = ['get_dataset']


def to_tens_and_norm(tr, mean, std,):
    return [tr.ToTensor(), tr.Normalize(mean=mean, std=std)]


def get_dataset(dataset, random_flip=False, color_jitter=False, gaussian_blur=False):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    dataset = partial(dataset, mean=mean, std=std)
    train_transform = [
        tr.RandomScale((0.7, 2)),
        tr.RandomCrop((512, 1024), pad_if_needed=True),
        tr.ToTensor(),
        tr.Normalize(mean=mean, std=std),
    ]

    test_transform = to_tens_and_norm(tr, mean, std)

    if random_flip:
        train_transform[0:0] = [tr.RandomHorizontalFlip(0.5)]
    if gaussian_blur:
        train_transform[-2:-2] = [tr.GaussianBlur()]
    if color_jitter:
        train_transform[-2:-2] = [tr.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)]

    train_transform = tr.Compose(train_transform)
    test_transform = tr.Compose(test_transform)

    return train_transform, test_transform, dataset
