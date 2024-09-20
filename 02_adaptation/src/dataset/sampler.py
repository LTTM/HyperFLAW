import random
import torch
from torch.utils.data import Sampler
from copy import deepcopy as cp
from typing import Iterator, List


class BatchSampler(Sampler[List[int]]):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
        shuffle (bool): If ``True``, the sampler will shuffle the batches.
        generator (Generator): Generator used for shuffling. If ``None``, then
            it will be initialized as ``torch.Generator()`` with a fixed seed of 0.

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, classes: dict, batch_size: int, drop_last: bool, shuffle: bool, generator: torch.Generator = None) -> None:
        if not isinstance(classes, dict):
            raise ValueError("classes should be a dict value, but got "
                             "classes={}".format(classes))
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        if not isinstance(shuffle, bool):
            raise ValueError("shuffle should be a boolean value, but got "
                             "shuffle={}".format(shuffle))

        self.classes = classes
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        if generator is not None:
            self.generator = generator
        else:
            self.generator = torch.Generator()
            self.generator.manual_seed(0)

        self.batches = []

    def __iter__(self) -> Iterator[List[int]]:
        return iter(self.batches)
    
    def __len__(self) -> int:
        return (len(self.batches) - 1) + (len(self.batches[-1]))
    
    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class SingleBatchSampler(BatchSampler):
    """
    Define a specific version of the BatchSampler for all the datasets that is
    composed by only one category of vehicles (only cars or only drones).

    Args:
        classes (dict): Dictionary with the classes of the dataset.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if its size would be less than ``batch_size``
        shuffle (bool): If ``True``, the sampler will shuffle the batches.
        generator (Generator): Generator used for shuffling. If ``None``, then it will be initialized as ``torch.Generator()`` with a fixed seed of 0.
        fuse_strategy (str): Strategy to fuse the batches. Can be one of ['alt', 'mix', 'rnd', 'og']. Default: 'og'.

            - 'alt': The batches are created alternating the classes (cars and drones).
            - 'mix': The batches are created mixing the classes (cars and drones) in each batch.
            - 'rnd': The batches are created randomply mixing the classes (cars and drones) in each batch (but not the weather).
            - 'og': The batches are created randomply mixing the classes (cars and drones) in each batch and also the weather.
    """

    def __init__(self, classes: dict, batch_size: int, drop_last: bool, shuffle: bool, generator: torch.Generator = None, fuse_strategy: str = 'og') -> None:

        super().__init__(classes, batch_size, drop_last, shuffle, generator)
        
        # Copy the classes
        cls = cp(self.classes)

        if fuse_strategy == 'alt':
            # Shuffle the classes
            if self.shuffle:
                for k in cls.keys():
                    perm = torch.randperm(len(cls[k]), generator=self.generator).tolist()
                    cls[k] = [cls[k][i] for i in perm]
                    # random.shuffle(cls[k])
            
            # Create the batches
            self.batches = []
            for k in cls.keys():
                self.batches += [cls[k][i:i+self.batch_size] for i in range(0, len(cls[k]), self.batch_size)]
        elif fuse_strategy == 'og':
            # Concatenate the self.classes dict in a single list
            cls_list = []
            for k in cls.keys():
                cls_list += cls[k]
            
            # Shuffle the list
            if self.shuffle:
                perm = torch.randperm(len(cls_list), generator=self.generator).tolist()
                cls_list = [cls_list[i] for i in perm]
            
            # Create the batches
            self.batches = [cls_list[i:i+self.batch_size] for i in range(0, len(cls_list), self.batch_size)]
        else:
            raise ValueError(f"fuse_strategy must be one of ['alt', 'og'], but got {fuse_strategy}")

        # Check the last batch to see if it should be dropped
        if self.drop_last:
            if len(self.batches[-1]) < self.batch_size:
                self.batches = self.batches[:-1]

        # Shuffle the batches
        if self.shuffle:
            perm = torch.randperm(len(self.batches), generator=self.generator).tolist()
            self.batches = [self.batches[i] for i in perm]
            # random.shuffle(self.batches)


class MixedBatchSampler(BatchSampler):
    """
    Define a specific version of the BatchSampler for all the dataset 
    that has both category of data cars and drones (only FLYSELMA).

    Args:
        classes (dict): Dictionary with the classes of the dataset.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if its size would be less than ``batch_size``
        shuffle (bool): If ``True``, the sampler will shuffle the batches.
        generator (Generator): Generator used for shuffling. If ``None``, then it will be initialized as ``torch.Generator()`` with a fixed seed of 0.
        fuse_strategy (str): Strategy to fuse the batches. Can be one of ['alt', 'mix', 'rnd', 'og']. Default: 'og'.

            - 'alt': The batches are created alternating the classes (cars and drones).
            - 'mix': The batches are created mixing the classes (cars and drones) in each batch.
            - 'rnd': The batches are created randomply mixing the classes (cars and drones) in each batch (but not the weather).
            - 'og': The batches are created randomply mixing the classes (cars and drones) in each batch and also the weather.
    """

    def __init__(self, classes: dict, batch_size: int, drop_last: bool, shuffle: bool, generator: torch.Generator = None, fuse_strategy = 'og') -> None:

        super().__init__(classes, batch_size, drop_last, shuffle, generator)
        
        # Copy the classes
        cls = cp(self.classes)

        # Shuffle the classes
        if self.shuffle:
            for k in cls.keys():
                for c in cls[k].keys():
                    perm = torch.randperm(len(cls[k][c]), generator=self.generator).tolist()
                    cls[k][c] = [cls[k][c][i] for i in perm]
        
        # Create the batches
        # if fuse_strategy is 'alt' then the batches are created alternating the classes (cars and drones)
        # if fuse_strategy is 'mix' then the batches are created mixing the classes (cars and drones) in each batch
        # if fuse_strategy is 'rnd' then the batches are created randomply mixing the classes (cars and drones) in each batch (but not the weather)
        # if fuse_strategy is 'og' then the batches are created randomply mixing the classes (cars and drones) in each batch and also the weather
        if self.batch_size == 1 and fuse_strategy == 'mix':
            fuse_strategy = 'alt'
        self.batches = []
        if fuse_strategy == 'alt':
            car_b = []
            drone_b = []
            car = cls['selma']
            drone = cls['flyawares']
            for c, d in zip(car.values(), drone.values()):
                car_b += [c[i:i+self.batch_size] for i in range(0, len(c), self.batch_size)]
                drone_b += [d[i:i+self.batch_size] for i in range(0, len(d), self.batch_size)]

            # Add the intermediate batches
            min_len = min(len(car_b), len(drone_b))
            for _ in range(min_len):
                self.batches += [car_b.pop(0)]
                self.batches += [drone_b.pop(0)]

            # Remove the last intermediate batch if it is not complete (only if it is not the last batch)
            if len(self.batches[-1]) < self.batch_size and (len(car_b) > 0 or len(drone_b) > 0):
                self.batches = self.batches[:-1]
            
            # Add the remaining batches (of the longest class)
            self.batches += car_b if len(car_b) > 0 else drone_b
        elif fuse_strategy == 'mix':
            car_b = []
            drone_b = []
            car = cls['selma']
            drone = cls['flyawares']
            for c, d in zip(car.values(), drone.values()):
                car_b += [c[i:i+self.batch_size//2] for i in range(0, len(c), self.batch_size//2)]
                drone_b += [d[i:i+self.batch_size//2] for i in range(0, len(d), self.batch_size//2)]

            # Add the intermediate batches
            min_len = min(len(car_b), len(drone_b))
            for _ in range(min_len):
                self.batches += [car_b.pop(0) + drone_b.pop(0)]

            # Remove the last intermediate batch if it is not complete (only if it is not the last batch)
            if len(self.batches[-1]) < self.batch_size and (len(car_b) > 0 or len(drone_b) > 0):
                self.batches = self.batches[:-1]

            for _ in range(len(car_b)//2):
                try:
                    self.batches += [car_b.pop(0) + car_b.pop(1)]
                except IndexError:
                    self.batches += [car_b.pop(0)]
            for _ in range(len(drone_b)//2):
                try:
                    self.batches += [drone_b.pop(0) + drone_b.pop(1)]
                except IndexError:
                    self.batches += [drone_b.pop(0)]
        elif fuse_strategy == 'rnd':
            for k in cls.keys():
                for c in cls[k].keys():
                    # Concatenate all the batches of the same weather
                    self.batches += [cls[k][c][i:i+self.batch_size] for i in range(0, len(cls[k][c]), self.batch_size)]

                    # Remove the last added batches if it is not complete (only if it is not the last batch)
                    if len(self.batches[-1]) < self.batch_size and k != list(cls.keys())[-1] and c != list(cls[k].keys())[-1]:
                        self.batches = self.batches[:-1]
        elif fuse_strategy == 'og':
            # Concatenate the self.classes dict in a single list
            cls_list = []
            for k in cls.keys():
                for c in cls[k].keys():
                    cls_list += cls[k][c]
            
            # Shuffle the list
            if self.shuffle:
                perm = torch.randperm(len(cls_list), generator=self.generator).tolist()
                cls_list = [cls_list[i] for i in perm]

            # Create the batches
            self.batches = [cls_list[i:i+self.batch_size] for i in range(0, len(cls_list), self.batch_size)]
        else:
            raise ValueError(f"fuse_strategy must be one of ['alt', 'mix', 'rnd'], but got {fuse_strategy}")

        # Check the last batch to see if it should be dropped
        if self.drop_last and len(self.batches[-1]) < self.batch_size:
                self.batches = self.batches[:-1]

        # Shuffle the batches
        if self.shuffle and fuse_strategy != 'alt':
            perm = torch.randperm(len(self.batches), generator=self.generator).tolist()
            self.batches = [self.batches[i] for i in perm]
