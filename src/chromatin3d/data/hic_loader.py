import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset

logger = logging.getLogger("chromatin3d")


def cut_hic(path, binsize, number_of_samples):
    p = Path(path).absolute()

    target_path = p.parent / p.stem
    target_path.mkdir(exist_ok=True)

    df = pd.read_table(p, header=None)
    df.columns = ["source", "target", "count"]

    rng = np.random.default_rng()

    for _ in range(number_of_samples):
        index = rng.integers(min(df["source"].min(), df["target"].min()), max(df["source"].max(), df["target"].max()) - 576 * binsize)

        frame = df[(df["source"] >= index) & (df["target"] <= index + 576 * binsize)]

        frame["source"] //= binsize
        frame["target"] //= binsize

        min_v = min(frame["source"].min(), frame["target"].min())

        frame["source"] -= min_v
        frame["target"] -= min_v

        result = np.zeros(shape=(576, 576), dtype=np.float32)

        for i in range(len(frame)):
            result[frame["source"].iloc[i], frame["target"].iloc[i]] = int(frame["count"].iloc[i])
            result[frame["target"].iloc[i], frame["source"].iloc[i]] = int(frame["count"].iloc[i])

        np.save(target_path / f"{p.stem}_{_}", result)

    return


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
