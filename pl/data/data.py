import sys

sys.path.append("/opt/ml/input/code/pl")

import pandas as pd
import pytorch_lightning as pl
import torch
import transformers
from tqdm.auto import tqdm

from utils.data_utils import *
from utils.utils import *
from datasets import load_from_disk

class Dataset(torch.utils.data.Dataset):
    """Dataset 구성을 위한 Class"""

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return item

    def __len__(self):
        return len(self.dataset)


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
            total_data = load_from_disk(self.train_path)

            train_data = total_data['train']
            val_data = total_data['validation']
            # tokenized_train = prepare_train_features(train_data, self.tokenizer)
            # tokenized_val = prepare_validation_features(val_data, self.tokenizer)
            tokenized_train = train_data.map(
            prepare_train_features,
            batched=True,
            num_proc=4,
            remove_columns=total_data['train'].column_names,
        )
            tokenized_val = val_data.map(
                prepare_validation_features,
                batched=True,
                num_proc=4,
                remove_columns=total_data['validation'].column_names)

            self.train_dataset = Dataset(tokenized_train)
            self.val_dataset = Dataset(tokenized_val)

        if stage == "test":
            # Test에 사용할 데이터를 호출 
            total_data = load_from_disk(self.train_path)

            train_data = total_data['train']
            val_data = total_data['validation']

            tokenized_val = prepare_validation_features(val_data, self.tokenizer)

            self.test_dataset = tokenized_val

        if stage == "predict":
            # Inference에 사용될 데이터를 호출
            p_data = load_from_disk(self.test_path)
            tokenized_p = prepare_validation_features(p_data, self.tokenizer)

            self.predict_dataset = tokenized_p

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