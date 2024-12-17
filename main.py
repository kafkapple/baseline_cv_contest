# main.py
import os
import torch
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import hydra

from data import data_prep, get_dataloaders, get_class_weights
from model_factory import get_model
from trainer import Trainer
from logger import get_logger
import torch.nn as nn
from torch.optim import Adam
from pathlib import Path

@hydra.main(version_base=None, config_path="./configs", config_name="config_base")
def main(cfg: DictConfig):
    # 주요 설정 출력
    data_path = Path(__file__).parent / cfg.data.path
    model_name = cfg.model.name
    print("\n=== Experiment Configuration ===")
    print(f"Data Path: {data_path}")
    print(f"Model: {model_name}")
    print(f"Data Split: {cfg.data.split_method}")
    print(f"Augmentation: {cfg.data.augmentation}")
    print(f"Class Balancing: {cfg.model.class_balancing.method}")
    print(f"Domain Adaptation: {cfg.model.domain_adaptation.method}")
    print(f"Learning Rate: {cfg.train.lr}")
    print(f"Batch Size: {cfg.train.batch_size}")
    print(f"Epochs: {cfg.train.epochs}")
    print(f"Image Size: {cfg.train.img_size}")
    print("==============================\n")

    
    num_classes = cfg.model.num_classes
    pretrained = cfg.model.pretrained
    img_size = cfg.train.img_size
    lr = cfg.train.lr
    epochs = cfg.train.epochs
    batch_size = cfg.train.batch_size
    num_workers = cfg.train.num_workers

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
### Data 준비
    try:
        data_prep(data_path, cfg)
    except Exception as e:
        print(f"데이터 준비 중 오류 발생: {str(e)}")
        return
    trn_loader, val_loader, tst_loader = get_dataloaders(data_path, cfg, batch_size=batch_size, 
                                             num_workers=num_workers, img_size=img_size)
### Model 준비
    model = get_model(model_name, num_classes=num_classes, pretrained=pretrained)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
### Logger 준비
    logger = get_logger(cfg)

    # 클래스 가중치 계산
    class_weights = None
    if cfg.model.class_balancing.method == "class_weight":
        train_df = pd.read_csv(os.path.join(data_path, "train_fold.csv"))
        class_weights = get_class_weights(train_df['target'].values)
        print("Class weights:", class_weights)

    # Train (validation 포함)
    trainer = Trainer(model, device, optimizer, loss_fn, 
                     logger=logger, class_weights=class_weights,
                     cfg=cfg)
    trainer.train(trn_loader, val_loader, epochs)

    # Inference
    preds_list = trainer.inference(tst_loader)

    sample_df = pd.read_csv(os.path.join(data_path, "sample_submission.csv"))
    sample_df['target'] = preds_list
    sample_df.to_csv("pred.csv", index=False)
    print(sample_df.head())

    if logger is not None:
        logger.finish()

def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

if __name__ == "__main__":
    main()
