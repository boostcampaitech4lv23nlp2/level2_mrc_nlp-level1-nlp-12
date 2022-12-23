import argparse
import os
import re

from datetime import datetime, timedelta

import pytorch_lightning as pl
import torch
import wandb

from datamodule.base_data import *
from models.base_model import *
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

time_ = datetime.now() + timedelta(hours=9)
time_now = time_.strftime("%m%d%H%M")

wandb_dict = {
    "gwkim_22": "f631be718175f02da4e2f651225fadb8541b3cd9",
    "rion_": "0d57da7f9222522c1a3dbb645a622000e0344d36",
    "daniel0801": "b8c4d92272716adcb1b2df6597cfba448854ff90",
    "seokhee": "c79d118b300d6cff52a644b8ae6ab0933723a59f",
    "dk100": "263b9353ecef00e35bdf063a51a82183544958cc",
}

if __name__ == "__main__":
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="sweep_config")
    args, _ = parser.parse_known_args()

    cfg = OmegaConf.load(f"/opt/ml/input/code/pl/config/{args.config}.yaml")
    # os.environ["WANDB_API_KEY"] = wandb_dict[cfg.wandb.wandb_username]
    wandb.login(key=wandb_dict[cfg.wandb.wandb_username])
    model_name_ch = re.sub("/", "_", cfg.model.model_name)
    pl.seed_everything(cfg.train.seed, workers=True)

    sweep_config = {
        "method": cfg.sweep.method,
        "parameters": {
            "lr": {
                "distribution": "uniform",  # parameter를 설정하는 기준을 선택합니다. uniform은 연속적으로 균등한 값들을 선택합니다.
                "min": cfg.sweep.lr_min,  # 최소값을 설정합니다.
                "max": cfg.sweep.lr_max,  # 최대값을 설정합니다.
            },
            "batch_size": {"values": [8, 16]},
        },
        "name": cfg.sweep.name,
        "metric": {"name": "val_em", "goal": "maximize"},
        "entity": cfg.wandb.wandb_entity,
        "project": cfg.wandb.wandb_project,
    }

    checkpoint_callback = ModelCheckpoint(
        monitor="val_em", save_top_k=1, save_last=True, save_weights_only=True, verbose=False, mode="max"
    )

    # Earlystopping
    earlystopping = EarlyStopping(monitor="val_em", patience=3, mode="max")
    # dataloader와 model을 생성합니다.
    def sweep_train(config=None):
        wandb.init(config=config)
        config = wandb.config

        dataloader = Dataloader(
            cfg.model.model_name,
            config.batch_size,
            cfg.data.shuffle,
            cfg.path.train_path,
            cfg.train.seed,
            cfg.path.test_path,
        )
        cfg.optimizer.learning_rate = config.lr
        cfg.train.batch_size = config.batch_size
        model = Model(cfg)
        wandb_logger = WandbLogger(
            log_model="all",
            name=f"{cfg.model.saved_name}_{cfg.train.batch_size}_{cfg.optimizer.learning_rate}_{time_now}",
            project=cfg.wandb.wandb_project,
            entity=cfg.wandb.wandb_entity,
        )
        trainer = pl.Trainer(
            gpus=1,
            max_epochs=10,
            log_every_n_steps=5,
            logger=wandb_logger,
            callbacks=[checkpoint_callback, earlystopping],
        )

        trainer.fit(model=model, datamodule=dataloader)
        trainer.test(model=model, datamodule=dataloader)
        ck_dir_path = f"/opt/ml/input/code/pl/checkpoint/{model_name_ch}"
        if not os.path.exists(ck_dir_path):
            os.makedirs(ck_dir_path)

        output_path = os.path.join(ck_dir_path, f"{model_name_ch}_{config.batch_size}_{config.lr}_{time_now}_model.pt")
        torch.save(model, output_path)

    sweep_id = wandb.sweep(
        sweep=sweep_config,  # config 딕셔너리를 추가합니다.
        project=cfg.wandb.wandb_project,  # project의 이름을 추가합니다.
    )
    wandb.agent(
        sweep_id=sweep_id,  # sweep의 정보를 입력하고
        function=sweep_train,  # train이라는 모델을 학습하는 코드를
        count=cfg.sweep.count,  # 총 5회 실행해봅니다.
    )
