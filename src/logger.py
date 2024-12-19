# logger.py
from omegaconf import DictConfig, OmegaConf
import wandb
from pathlib import Path
from datetime import datetime
import json
import os
import torch

class Logger:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # 실험 시작 시간으로 실험 ID 생성
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # wandb 초기화
        self.use_wandb = cfg.logger.use_wandb
        if self.use_wandb:
            # 주요 설정으로 run name 구성
            run_name = f"{cfg.model.name}_{cfg.train.img_size}_{cfg.data.split_method}"
            if cfg.model.domain_adaptation.method != "none":
                run_name += f"_{cfg.model.domain_adaptation.method}"
            run_name += f"_{self.experiment_id}"
            
            # 주요 설정으로 tags 구성
            tags = [
                f"model_{cfg.model.name}",
                f"img_size_{cfg.train.img_size}",
                f"split_{cfg.data.split_method}",
                f"aug_{cfg.data.augmentation}",
                f"batch_{cfg.train.batch_size}",
                f"lr_{cfg.train.lr}",
                f"da_{cfg.model.domain_adaptation.method}"
            ]
            
            # wandb config에 전체 epochs 수 포함
            config = OmegaConf.to_container(cfg, resolve=True)
            config['total_epochs'] = cfg.train.epochs
            
            # wandb 초기화
            wandb.init(
                entity=cfg.logger.entity,
                project=cfg.logger.project_name,
                name=run_name,
                tags=tags,
                config=config,
            )
            
            # 메트릭 정의 (init 이후에)
            wandb.define_metric("train/loss", summary="min")
            wandb.define_metric("train/accuracy", summary="max")
            wandb.define_metric("train/f1", summary="max")
            wandb.define_metric("val/loss", summary="min")
            wandb.define_metric("val/accuracy", summary="max")
            wandb.define_metric("val/f1", summary="max")
            
            # 클래스별 F1 메트릭 정의
            for i in range(cfg.model.num_classes):
                wandb.define_metric(f"train/f1_class_{i}", summary="max")
                wandb.define_metric(f"val/f1_class_{i}", summary="max")
        
        # 실험 설정 저장 및 출력
        self.save_config()
        self.print_experiment_info()
    
    def print_experiment_info(self):
        """실험 설정 출력"""
        print("\n=== Experiment Configuration ===")
        print(f"Experiment ID: {self.experiment_id}")
        print(f"Data Path: {self.cfg.data.path}")
        print(f"Model: {self.cfg.model.name}")
        print("\n[Data Settings]")
        print(f"Split Method: {self.cfg.data.split_method}")
        print(f"Validation Size: {self.cfg.data.val_size}")
        print(f"Augmentation: {self.cfg.data.augmentation}")
        
        print("\n[Model Settings]")
        print(f"Pretrained: {self.cfg.model.pretrained}")
        print(f"Domain Adaptation: {self.cfg.model.domain_adaptation.method}")
        
        print("\n[Training Settings]")
        print(f"Learning Rate: {self.cfg.train.lr}")
        print(f"Batch Size: {self.cfg.train.batch_size}")
        print(f"Epochs: {self.cfg.train.epochs}")
        print(f"Image Size: {self.cfg.train.img_size}")
        print(f"Number of Workers: {self.cfg.train.num_workers}")
        print("==============================\n")
    
    def save_config(self):
        """실험 설정을 파일로 저장"""
        config_path = self.log_dir / f"config_{self.experiment_id}.json"
        config_dict = OmegaConf.to_container(self.cfg, resolve=True)
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def log_metrics(self, metrics: dict, step: int = None, phase: str = None):
        """메트릭 로깅"""
        if self.use_wandb:
            wandb_metrics = {}
            
            # 주요 메트릭
            main_keys = ['loss', 'accuracy', 'f1']
            for k in main_keys:
                if k in metrics:
                    wandb_metrics[f"{phase}/{k}"] = metrics[k]
            
            # 클래스별 F1
            for k, v in metrics.items():
                if k.startswith('f1_class_'):
                    wandb_metrics[f"{phase}/{k}"] = v
            
            # 로깅
            wandb.log(wandb_metrics, step=step)
        
        # 콘솔 출력 - 주요 메트릭만 출력
        if phase:
            print(f"\n[{phase.upper()} Epoch {step+1 if step is not None else ''}]")
            metrics_str = []
            for k in ['loss', 'accuracy', 'f1']:
                if k in metrics:
                    metrics_str.append(f"{k}: {metrics[k]:.4f}")
            print(" | ".join(metrics_str))
    
    def log_batch(self, metrics: dict, epoch: int, batch: int, num_batches: int, phase: str = "train"):
        """배치 단위 로깅 - wandb에만 기록"""
        if self.use_wandb:
            wandb_metrics = {f"{phase}/{k}": v for k, v in metrics.items()}
            wandb.log(wandb_metrics)
    
    def save_model(self, model, optimizer, epoch: int, metrics: dict = None):
        """모델 체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        
        save_path = self.log_dir / f"checkpoint_{self.experiment_id}_epoch{epoch+1}.pt"
        torch.save(checkpoint, save_path)
        print(f"\nCheckpoint saved: {save_path}")
    
    def finish(self):
        """실험 종료"""
        if self.use_wandb:
            wandb.finish()
        print("\nExperiment finished!")

def get_logger(cfg: DictConfig) -> Logger:
    """Logger 인스턴스를 생성하고 반환하는 함수"""
    # wandb 관련 환경 변수 설정
    os.environ["WANDB_SILENT"] = "true"
    
    # 로그 디렉토리 생성
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    return Logger(cfg)
