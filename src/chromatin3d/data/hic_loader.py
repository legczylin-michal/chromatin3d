import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset
import torch

import logging

logger = logging.getLogger("chromatin3d")


def cut_hic(path, binsize, number_of_samples):
    pass


class HiCDataset(Dataset):
    def __init__(self, p_root_path):
        self.__root_path = Path(p_root_path).absolute()

        self.__filepaths = []
        if self.__root_path.is_dir():
            self.__filepaths = list(self.__root_path.iterdir())

        return

    def __len__(self):
        return len(self.__filepaths)

    def __getitem__(self, index: int):
        return torch.tensor(np.load(self.__filepaths[index]))


def split(p_root_path, random_state=None):
    dataset = HiCDataset(p_root_path)

    indices = list(range(len(dataset)))

    train_indices, valid_indices = train_test_split(indices, train_size=0.7, random_state=random_state, shuffle=True)

    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, valid_indices)

    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=10, shuffle=True)

    return train_dataloader, valid_dataloader
