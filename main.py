# main.py
import os
import torch
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import hydra
from torch.optim import Adam

from src.data import data_prep, get_dataloaders
from src.models import get_model
from src.trainer import Trainer
from src.logger import get_logger
import torch.nn as nn
from pathlib import Path

def set_seed(seed: int):
    """모든 random seed를 고정"""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(cfg: DictConfig):
    # Global seed 설정
    set_seed(cfg.seed)
    
    # Logger 초기화 (실험 설정 출력 포함)
    logger = get_logger(cfg)
    
    # 설정값 추출
    data_path = Path(__file__).parent / cfg.data.path
    output_dir = Path(__file__).parent / cfg.output_dir
    output_dir.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data 준비
    try:
        data_prep(data_path, cfg)
    except Exception as e:
        logger.log_metrics({"error": str(e)}, phase="data_prep")
        return
    
    trn_loader, val_loader, tst_loader = get_dataloaders(data_path, cfg)

    # Model 준비
    model = get_model(
        cfg.model.name, 
        num_classes=cfg.model.num_classes,
        pretrained=cfg.model.pretrained,
        cfg=cfg
    )
    model = model.to(device)
    optimizer = Adam(
        model.parameters(), 
        lr=cfg.train.lr,
        weight_decay=cfg.model.regularization.weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()

    # Train (validation 포함)
    trainer = Trainer(model, device, optimizer, loss_fn, logger=logger, cfg=cfg)
    trainer.train(trn_loader, val_loader, cfg.train.epochs)

    # Inference
    preds_list = trainer.inference(tst_loader)

    # 예측 결과 저장
    sample_df = pd.read_csv(data_path / "sample_submission.csv")
    sample_df['target'] = preds_list
    pred_path = output_dir / f"{logger.experiment_id}.csv"
    sample_df.to_csv(pred_path, index=False)
    logger.log_metrics({"predictions_saved": str(pred_path)}, phase="inference")

    # 실험 종료
    logger.finish()

if __name__ == "__main__":
    main()
