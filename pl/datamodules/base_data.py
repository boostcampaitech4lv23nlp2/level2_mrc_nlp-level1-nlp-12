import sys

sys.path.append("/opt/ml/input/code/pl")

import pandas as pd
import pytorch_lightning as pl
import torch
import transformers
from tqdm.auto import tqdm

from input.code.pl.utils.util import *


class Dataset(torch.utils.data.Dataset):
    """Dataset 구성을 위한 Class"""

    def __init__(self, pair_dataset, labels):
        self.pair_dataset = pair_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: val[idx].clone().detach() for key, val in self.pair_dataset.items()
        }
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class Dataloader(pl.LightningDataModule):
    """ 
    Trainer에 들어갈 데이터셋을 호출
    """

    def __init__(
        self, model_name, batch_size, shuffle, train_path, test_path, split_seed=42
    ):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.split_seed = split_seed

        self.train_path = train_path
        self.test_path = test_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, max_length=200
        )

    def setup(self, stage="fit"):
        if stage == "fit":
            # 학습 데이터을 호출
            total_data = load_data(self.train_path)

            train_data = total_data.sample(frac=0.9, random_state=self.split_seed)
            val_data = total_data.drop(train_data.index)

            train_label = label_to_num(train_data["label"].values)
            val_label = label_to_num(val_data["label"].values)

            tokenized_train = tokenized_dataset(train_data, self.tokenizer)
            tokenized_val = tokenized_dataset(val_data, self.tokenizer)

            self.train_dataset = Dataset(tokenized_train, train_label)
            self.val_dataset = Dataset(tokenized_val, val_label)

        if stage == "test":
            # Test에 사용할 데이터를 호출
            total_data = load_data(self.train_path)

            train_data = total_data.sample(frac=0.9, random_state=self.split_seed)
            val_data = total_data.drop(train_data.index)

            val_label = label_to_num(val_data["label"].values)
            tokenized_val = tokenized_dataset(val_data, self.tokenizer)

            self.test_dataset = Dataset(tokenized_val, val_label)

        if stage == "predict":
            # Inference에 사용될 데이터를 호출
            p_data = load_data(self.test_path)
            p_label = list(map(int, p_data["label"].values))
            tokenized_p = tokenized_dataset(p_data, self.tokenizer)

            self.predict_dataset = Dataset(tokenized_p, p_label)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=4,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=4
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=4
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.predict_dataset, batch_size=self.batch_size, num_workers=4
        )
