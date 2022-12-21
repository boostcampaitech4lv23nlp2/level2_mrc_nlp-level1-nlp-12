import sys
from importlib import import_module

sys.path.append("/opt/ml/input/code/level2_mrc_nlp-level1-nlp-12/pl")
import numpy as np
import pytorch_lightning as pl
import torch
import transformers

from utils.utils import (criterion_entrypoint, compute_metrics)
from utils.data_utils import *
from datasets import load_from_disk

class Model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = config.model.model_name
        self.lr = config.optimizer.learning_rate
        self.lr_sch_use = config.optimizer.lr_sch_use
        self.lr_decay_step = config.optimizer.lr_decay_step
        self.scheduler_name = config.optimizer.scheduler_name
        self.lr_weight_decay = config.optimizer.lr_weight_decay

        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForQuestionAnswering.from_pretrained(
            pretrained_model_name_or_path=self.model_name
        )
        # Loss 계산을 위해 사용될 CE Loss를 호출합니다.
        self.loss_func = criterion_entrypoint(config.loss.loss_name)
        self.optimizer_name = config.optimizer.optimizer_name
        
        self.eval_dataset = load_from_disk(config.path.train_path)['validation']
        self.predict_dataset = load_from_disk(config.path.test_path)['validation']
        
    def forward(self, x):
        x = self.plm(
            input_ids=x["input_ids"],
            attention_mask=x["attention_mask"],
            # token_type_ids=x["token_type_ids"],
        )
        return x['start_logits'], x['end_logits']

    def training_step(self, batch):
        output = self(batch)
        start_logits, end_logits = output
        s_position, e_position = batch['start_positions'], batch['end_positions']
        
        l_s = self.loss_func(start_logits, s_position)
        l_e = self.loss_func(end_logits, e_position)

        loss = (l_s+l_e) / 2
        self.log("train_loss", loss)
        
        return loss

    def validation_step(self, batch, batch_idx):
        data, id = batch
        output = self(data)
        start_logits, end_logits = output
        # prediction = (start_logits.argmax(dim=-1), end_logits.argmax(dim=-1))
        prediction = (start_logits, end_logits)
        # print(start_logits.shape)
        # print(self.eval_dataset[batch_idx], data.keys())
        preds = post_processing_function(self.eval_dataset, data, id, prediction, 'eval')
        result = compute_metrics(preds)
        #print(preds.predictions, preds.label_ids)
        #print(result)
        self.log("val_em", result['exact_match'])
        self.log("val_f1", result['f1'])

    def test_step(self, batch, batch_idx):
        data, id = batch
        start_logits, end_logits = self(data).split(1, dim=-1)
        # prediction = (start_logits.argmax(dim=-1), end_logits.argmax(dim=-1))
        prediction = (start_logits, end_logits)
        preds = post_processing_function(self.eval_dataset, data, id, prediction, 'eval')
        result = compute_metrics(preds)
        self.log("test_em", result['exact_match'])
        self.log("test_f1", result['f1'])

    # def predict_step(self, batch, batch_idx):

    #     start_logits, end_logits = self(batch)

    #     preds = post_processing_function(self.predict_dataset[batch_idx], batch, logits, 'predict')

    #     return preds

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