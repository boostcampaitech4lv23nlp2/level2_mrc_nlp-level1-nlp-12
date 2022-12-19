import sys
from importlib import import_module

sys.path.append("/opt/ml/input/code/pl")
import numpy as np
import pytorch_lightning as pl
import torch
import transformers

from utils import (criterion_entrypoint, klue_re_auprc, klue_re_micro_f1,
                   n_compute_metrics)


class Model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = config.model.model_name
        self.lr = config.train.learning_rate
        self.lr_sch_use = config.train.lr_sch_use
        self.lr_decay_step = config.train.lr_decay_step
        self.scheduler_name = config.train.scheduler_name
        self.lr_weight_decay = config.train.lr_weight_decay

        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.model_name, num_labels=30
        )
        # Loss 계산을 위해 사용될 CE Loss를 호출합니다.
        self.loss_func = criterion_entrypoint(config.train.loss_name)
        self.optimizer_name = config.train.optimizer_name

    def forward(self, x):
        x = self.plm(
            input_ids=x["input_ids"],
            attention_mask=x["attention_mask"],
            token_type_ids=x["token_type_ids"],
        )
        return x["logits"]

    def training_step(self, batch, batch_idx):
        x = batch
        y = batch["labels"]

        logits = self(x)
        loss = self.loss_func(logits, y.long())

        f1, accuracy = n_compute_metrics(logits, y).values()
        self.log("train", {"loss": loss, "f1": f1, "accuracy": accuracy})

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        y = batch["labels"]

        logits = self(x)
        loss = self.loss_func(logits, y.long())

        f1, accuracy = n_compute_metrics(logits, y).values()
        self.log("val_loss", loss)
        self.log("val_accuracy", accuracy)
        self.log("val_f1", f1, on_step=True)

        return {"logits": logits, "y": y}

    def validation_epoch_end(self, outputs):
        logits = torch.cat([x["logits"] for x in outputs])
        y = torch.cat([x["y"] for x in outputs])

        logits = logits.detach().cpu().numpy()
        y = y.detach().cpu()

        auprc = klue_re_auprc(logits, y)
        self.log("val_auprc", auprc)

    def test_step(self, batch, batch_idx):
        x = batch
        y = batch["labels"]

        logits = self(x)

        f1, accuracy = n_compute_metrics(logits, y).values()
        self.log("test_f1", f1)

        return {"logits": logits, "y": y}

    def test_epoch_end(self, outputs):
        logits = torch.cat([x["logits"] for x in outputs])
        y = torch.cat([x["y"] for x in outputs])

        logits = logits.detach().cpu().numpy()
        y = y.detach().cpu()

        auprc = klue_re_auprc(logits, y)
        self.log("test_auprc", auprc)

    def predict_step(self, batch, batch_idx):
        logits = self(batch)

        return logits

    def configure_optimizers(self):
        opt_module = getattr(import_module("torch.optim"), self.optimizer_name)
        if self.lr_weight_decay:
            optimizer = opt_module(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.lr,
                weight_decay=0.01,
            )
        else:
            optimizer = opt_module(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.lr,
                # weight_decay=5e-4
            )
        if self.lr_sch_use:
            t_total = 2030 * 7  # train_dataloader len, epochs
            warmup_step = int(t_total * 0.1)

            _scheduler_dic = {
                "StepLR": torch.optim.lr_scheduler.StepLR(
                    optimizer, self.lr_decay_step, gamma=0.5
                ),
                "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, factor=0.1, patience=10
                ),
                "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=2, eta_min=0.0
                ),
                "constant_warmup": transformers.get_constant_schedule_with_warmup(
                    optimizer, 100
                ),
                "cosine_warmup": transformers.get_cosine_schedule_with_warmup(
                    optimizer, num_warmup_steps=10, num_training_steps=t_total
                ),
            }
            scheduler = _scheduler_dic[self.scheduler_name]

            return [optimizer], [scheduler]
        else:
            return optimizer
