import sys

sys.path.append("/opt/ml/input/code/pl")

from functools import partial

import pandas as pd
import pytorch_lightning as pl
import torch
import transformers

from datasets import load_from_disk
from tqdm.auto import tqdm
from utils.data_utils import *
from utils.util import *

class Train_Dataset(torch.utils.data.Dataset):
    """Dataset 구성을 위한 Class"""

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        # print(self.dataset[idx])
        item = {key: torch.tensor(val) for key, val in self.dataset[idx].items()}
        return item

    def __len__(self):
        return len(self.dataset)


class Val_Dataset(torch.utils.data.Dataset):
    """Dataset 구성을 위한 Class"""

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.dataset[idx]["input_ids"]),
            "attention_mask": torch.tensor(self.dataset[idx]["attention_mask"]),
            "offset_mapping": torch.tensor(self.dataset[idx]["offset_mapping"]),
        }
        id = self.dataset[idx]["example_id"]

        return item, id

    def __len__(self):
        return len(self.dataset)


class Dataloader(pl.LightningDataModule):
    """
    Trainer에 들어갈 데이터셋을 호출
    """

    def __init__(self, model_name, batch_size, shuffle, train_path, test_path, split_seed, retrieval):
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

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, max_length=200)
        self.retrieval = retrieval

    def setup(self, stage="fit"):
        if stage == "fit":
            # 학습 데이터을 호출
            total_data = load_from_disk(self.train_path)
            tokenized_train = total_data.map(
                prepare_train_features,
                batched=True,
                num_proc=4,
                remove_columns=total_data["train"].column_names,
                fn_kwargs={"tokenizer": self.tokenizer},
            )
            tokenized_val = total_data.map(
                prepare_validation_features,
                batched=True,
                num_proc=4,
                remove_columns=total_data["validation"].column_names,
                fn_kwargs={"tokenizer": self.tokenizer},
            )

            self.train_dataset = Train_Dataset(tokenized_train["train"])
            self.val_dataset = Val_Dataset(tokenized_val["validation"])

        if stage == "test":
            # Test에 사용할 데이터를 호출
            total_data = load_from_disk(self.train_path)

            tokenized_val = total_data.map(
                prepare_validation_features,
                batched=True,
                num_proc=4,
                remove_columns=total_data["validation"].column_names,
                fn_kwargs={"tokenizer": self.tokenizer},
            )

            self.test_dataset = Val_Dataset(tokenized_val["validation"])

        if stage == "predict":
            # Inference에 사용될 데이터를 호출
            dataset = load_from_disk(self.test_path)
            dataset = run_sparse_retrieval(self.tokenizer.tokenize, dataset, "predict", False, self.retrieval)
            column_names = dataset["validation"].column_names

            tokenized_p = dataset.map(
                prepare_validation_features,
                batched=True,
                num_proc=4,
                remove_columns=column_names,
                fn_kwargs={"tokenizer": self.tokenizer},
            )

            self.predict_dataset = Val_Dataset(tokenized_p["validation"])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=0,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=0,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=0,
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=0,
        )
