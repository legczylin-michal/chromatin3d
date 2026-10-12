import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset

logger = logging.getLogger("chromatin3d")


def cut_hic(path, binsize, number_of_samples):
    df = pd.read_table(path, header=None)

    result = {}

    for _, record in df.iterrows():
        i = record[0] // binsize
        j = record[1] // binsize

        row = result.get(i, {})
        row[j] = row.get(j, 0) + record[2]
        result[i] = row

    df2 = pd.DataFrame(result)

    print(df2)

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


if __name__ == "__main__":
    cut_hic("F:/_/D/SCIENCE/STUDIES/masters/thesis/chromatin3d/data/ENCFF216QQM/ENCFF216QQM_chr1_5kbp.txt", 5000, 100)
