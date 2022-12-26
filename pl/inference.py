import argparse

from datamodule.base_data import *
from utils.data_utils import *
from utils.util import *
from omegaconf import OmegaConf
from models.base_model import *
from retrievals.base_retrieval import SparseRetrieval
if __name__ == "__main__":
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="base_config")
    args, _ = parser.parse_known_args()

    cfg = OmegaConf.load(f"/opt/ml/code/pl/config/{args.config}.yaml")
    pl.seed_everything(cfg.train.seed, workers=True)


    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(
        cfg.model.model_name,
        cfg.train.batch_size,
        cfg.data.shuffle,
        cfg.path.train_path,
        cfg.path.test_path,
        cfg.train.seed,
    )

    ckpt_path = "/opt/ml/code/pl/checkpoint/klue_roberta-large/epoch=8_val_loss=0.5509.ckpt"
    pt_path = "/opt/ml/code/pl/output/klue_roberta-large_11302158_model.pt"

    # for checkpoint
    model = Model(cfg).load_from_checkpoint(checkpoint_path=ckpt_path)

    # for pt
    # model = Model(cfg)
    # model.load_state_dict(torch.load(pt_path))

    # gpu가 없으면 'gpus=0'을, gpu가 여러개면 'gpus=4'처럼 사용하실 gpu의 개수를 입력해주세요
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=cfg.train.max_epoch,
        log_every_n_steps=cfg.train.logging_step,
        deterministic=True,
    )

    predictions = trainer.predict(model=model, datamodule=dataloader)

    # predict한 logits값으로 submission.csv를 만듭니다.
    make_output(logits)

    print("---- Finish ----")
