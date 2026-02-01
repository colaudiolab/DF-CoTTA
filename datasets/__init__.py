# -*- coding: utf-8 -*-

from .DatasetWrapper import DatasetWrapper
from .MNIST import MNIST_ as MNIST
from .CIFAR import CIFAR10_ as CIFAR10
from .CIFAR import CIFAR100_ as CIFAR100
from .ImageNet import ImageNet_ as ImageNet
from typing import Union
from .Features import Features
from .FAV_DIL import Multimodal_dataset as FAV
from .MDCDDataset import MDCDDataset


__all__ = [
    "load_dataset",
    "dataset_list",
    "MNIST",
    "CIFAR10",
    "CIFAR100",
    "ImageNet",
    "DatasetWrapper",
    "Features",
    "FAV",
]

dataset_list = {
    # "MNIST": MNIST,
    # "CIFAR-10": CIFAR10,
    # "CIFAR-100": CIFAR100,
    "multidataset": FAV,
    "FakeAVCeleb": FAV,
    "multidataset_imbalance_1_5": FAV,
    "multidataset_imbalance_1_10": FAV,
    "multidataset_imbalance_1_20": FAV,
    "multidataset_imbalance_1_40": FAV,
    "multidataset_imbalance_2_5_10_20": FAV,
    "multidataset_imbalance_2_5_10_20_re": FAV,
    'MDCDDataset': MDCDDataset,
}


def load_dataset(
    name: str,
    root: str,
    train: bool,
    base_ratio: float,
    num_phases: int,
    augment: bool = False,
    inplace_repeat: int = 1,
    shuffle_seed: int | None = None,
    *args,
    **kwargs
# ) -> Union[FAV]:
) -> Union[MNIST, CIFAR10, CIFAR100, ImageNet, FAV]:
    return dataset_list[name](
        root=root,
        train=train,
        base_ratio=base_ratio,
        num_phases=num_phases,
        # augment=augment,
        inplace_repeat=inplace_repeat,
        # shuffle_seed=shuffle_seed,
        *args,
        **kwargs
    )

class DataManager():
    def __init__(self, name: str,
    root: str,
    base_ratio: float,
    num_phases: int,
    augment: bool = False,
    inplace_repeat: int = 1,
    shuffle_seed: int | None = None,
    *args,
    **kwargs):
        self.name = name
        self.root = root
        self.base_ratio = base_ratio
        self.num_phases = num_phases
        self.augment = augment
        self.inplace_repeat = inplace_repeat
        self.shuffle_seed = shuffle_seed
        self.args = args
        self.kwargs = kwargs
        self.train_dataset = load_dataset(self.name, self.root, True, self.base_ratio, self.num_phases, False, self.inplace_repeat, self.shuffle_seed, **self.kwargs)
        self.test_dataset = load_dataset(self.name, self.root, False, self.base_ratio, self.num_phases, False, self.inplace_repeat, self.shuffle_seed, **self.kwargs)
    def get_dataset(self, train: bool):
        if train:
            # dataset = load_dataset(train=True, augment=False, **self.kwargs)
            dataset = self.train_dataset
        else:
            dataset = self.test_dataset
        return dataset
        