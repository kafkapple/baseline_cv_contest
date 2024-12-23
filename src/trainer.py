# trainer.py
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
import numpy as np
from typing import Dict, Any, Optional
import torch.nn.functional as F
import os
import random
import datetime
from pathlib import Path
import pandas as pd

class Trainer:
    def __init__(self, model, device, optimizer, criterion, logger, cfg):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.logger = logger
        self.cfg = cfg
        self.epochs = cfg.train.epochs
        
        # Mixed Precision 설정
        self.use_amp = cfg.train.mixed_precision.enabled
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Early Stopping 설정
        self.early_stopping = cfg.train.early_stopping.enabled
        if self.early_stopping:
            self.patience = cfg.train.early_stopping.get('patience', 5)  # 기본값 5
            self.min_delta = cfg.train.early_stopping.get('min_delta', 0.001)  # 기본값 0.001
            self.best_f1 = 0.0  # best_val_loss 대신 best_f1 사용
            self.patience_counter = 0

        if cfg.model.regularization.label_smoothing > 0:
            self.criterion = nn.CrossEntropyLoss(
                label_smoothing=cfg.model.regularization.label_smoothing
            )
        else:
            self.criterion = criterion

        self.best_f1 = 0.0
        self.best_metric = {'f1': 0.0}  # 추가

    def train_epoch(self, train_loader, epoch):
        # DataLoader worker의 seed 설정
        def seed_worker(worker_id):
            worker_seed = self.cfg.seed + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        
        g = torch.Generator()
        g.manual_seed(self.cfg.seed)
        
        train_loader.worker_init_fn = seed_worker
        train_loader.generator = g
        
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=True)
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            
            # Domain Adaptation 사용 시
            if hasattr(self.model, 'method') and self.model.method == "dann":
                # Progressive alpha for DANN
                p = float(batch_idx + epoch * len(train_loader)) / (self.epochs * len(train_loader))
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
                
                # Forward pass with domain adaptation
                class_output, domain_output = self.model(data, alpha)
                
                # Domain labels (0: source, 1: target)
                domain_labels = torch.zeros(data.size(0), dtype=torch.long).to(self.device)
                
                # Calculate losses
                class_loss = self.criterion(class_output, target)
                domain_loss = F.cross_entropy(domain_output, domain_labels)
                loss = class_loss + domain_loss
            else:
                # 기존 학습 로직
                output = self.model(data)
                loss = self.criterion(output, target)
            
            loss.backward()
            self.optimizer.step()
            
            # 메트릭 계산
            preds = output.argmax(dim=1).cpu().numpy()
            labels = target.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
            
            total_loss += loss.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # 에폭 단위 메트릭 계산
        epoch_metrics = {
            'loss': total_loss / len(train_loader),
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds, average='macro')
        }
        
        # 클래스별 F1 추가
        class_f1 = f1_score(all_labels, all_preds, average=None)
        for i, f1 in enumerate(class_f1):
            epoch_metrics[f'f1_class_{i}'] = f1
        
        return epoch_metrics

    def validate(self, val_loader):
        """Validation 행"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc='Validation', leave=True):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                preds = output.argmax(dim=1).cpu().numpy()
                labels = target.cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)
                
                total_loss += loss.item()
        
        # 1. 기본 메트릭 계산
        val_metrics = {
            'loss': total_loss / len(val_loader),
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds, average='macro')
        }
        
        # 2. 클래스별 성능 분석
        class_f1 = f1_score(all_labels, all_preds, average=None)
        for i, f1 in enumerate(class_f1):
            val_metrics[f'f1_class_{i}'] = f1
        
        # 3. 클래스�� confusion matrix (옵션)
        if hasattr(self.cfg.logger, 'save_confusion_matrix') and self.cfg.logger.save_confusion_matrix:
            cm = confusion_matrix(all_labels, all_preds)
            val_metrics['confusion_matrix'] = cm
        
        return val_metrics

    def train(self, train_loader, val_loader, epochs):
        for epoch in range(epochs):
            # 학습
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # 검증
            val_metrics = None
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                
                # Best model 판단
                current_f1 = val_metrics['f1']
                if current_f1 > self.best_f1:
                    self.best_f1 = current_f1
                    self.patience_counter = 0
                    self.save_checkpoint(epoch, is_best=True)
                else:
                    self.patience_counter += 1
            
            # 한번에 메트릭 출력
            self._print_metrics(epoch, epochs, train_metrics, val_metrics)
            
            # W&B 로깅
            self.logger.log_metrics(train_metrics, step=epoch, phase="train")
            if val_metrics:
                self.logger.log_metrics(val_metrics, step=epoch, phase="val")

    def inference(self, test_loader):
        # best model 로드
        checkpoint = torch.load(Path(self.cfg.output_dir) / "checkpoints" / "best.pt")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 이후 inference 진행
        self.model.eval()
        all_preds = []
        
        with torch.no_grad():
            for data, _ in tqdm(test_loader, desc="Inference"):
                data = data.to(self.device)
                output = self.model(data, mode='main')
                preds = output.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
        
        # 예측 결과 저장
        predictions_df = pd.DataFrame({
            'ID': test_loader.dataset.df['ID'],
            'target': all_preds
        })
        self.save_predictions(predictions_df)
        
        return all_preds

    def update_bn_stats(self, loader):
        """AdaBN: 테스트 데이터로 BatchNorm 통계 업데이트"""
        self.model.train()  # BN 통계를 업데이트하기 위해 train 모드로 설정
        with torch.no_grad():  # 그래디언트는 계산하지 않음
            for images, _ in tqdm(loader, desc="Updating BN statistics"):
                images = images.to(self.device)
                _ = self.model(images)  # forward pass만 수행하여 BN ���계 업데이트

    def adapt_test_time(self, loader):
        """Test-Time Adaptation using rotation prediction"""
        self.model.train()  # BN 통계 업데이트 해
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        
        with tqdm(loader, desc="Test-Time Adaptation") as pbar:
            for images, _ in pbar:
                images = images.to(self.device)
                batch_size = images.size(0)
                
                # 4가지 회전 버전 생성 (0, 90, 180, 270도)
                rotated_images = []
                rotation_labels = []
                
                for k in range(4):
                    rotated = torch.rot90(images, k, [2, 3])  # 이미지 회전
                    rotated_images.append(rotated)
                    rotation_labels.extend([k] * batch_size)
                
                rotated_images = torch.cat(rotated_images, dim=0)
                rotation_labels = torch.tensor(rotation_labels).to(self.device)
                
                # 회전 예측
                optimizer.zero_grad()
                rotation_preds = self.model(rotated_images, mode='rotation')
                loss = F.cross_entropy(rotation_preds, rotation_labels)
                
                loss.backward()
                optimizer.step()
                
                pbar.set_postfix({'rot_loss': f'{loss.item():.4f}'})

    def _calculate_metrics(self, preds_array: np.ndarray, targets_array: np.ndarray, loss: float, prefix: str = "train") -> Dict[str, Any]:
        """공통 메트릭 계산 함수"""
        metrics = {
            f"{prefix}/loss": loss,
            f"{prefix}/accuracy": accuracy_score(targets_array, preds_array),
            f"{prefix}/f1": f1_score(targets_array, preds_array, average='macro')
        }
        
        # 클래스별 F1 계산
        if self.logger is not None:
            class_f1 = f1_score(targets_array, preds_array, average=None)
            for i, f1 in enumerate(class_f1):
                metrics[f"{prefix}/f1_class_{i}"] = f1
        
        return metrics

    def _print_metrics(self, epoch: int, total_epochs: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """메트릭 출력 함수"""
        print(f"\nEpoch {epoch+1}/{total_epochs}")
        
        # 학습 메트릭 출력
        print(f"Train - Loss: {train_metrics['train/loss']:.4f}, "
              f"Acc: {train_metrics['train/accuracy']:.4f}, "
              f"F1: {train_metrics['train/f1']:.4f}")
        
        # 검증 메트릭 출력 (있는 경우)
        if val_metrics:
            print(f"Val   - Loss: {val_metrics['val/loss']:.4f}, "
                  f"Acc: {val_metrics['val/accuracy']:.4f}, "
                  f"F1: {val_metrics['val/f1']:.4f}")

            print()
        

    def _calculate_class_metrics(self, preds, labels):
        """클래스별 세부 메트릭 계산"""
        metrics = {}
        
        # 클래스별 F1 score
        class_f1 = f1_score(labels, preds, average=None)
        for i, f1 in enumerate(class_f1):
            metrics[f'f1_class_{i}'] = f1
        
        # 클래스별 precision/recall
        class_precision = precision_score(labels, preds, average=None)
        class_recall = recall_score(labels, preds, average=None)
        
        for i, (prec, rec) in enumerate(zip(class_precision, class_recall)):
            metrics[f'precision_class_{i}'] = prec
            metrics[f'recall_class_{i}'] = rec
        
        return metrics

    def save_checkpoint(self, epoch, is_best=False):
        # timestamp 기반 디렉토리 생성
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(self.cfg.output_dir) / "checkpoints" / timestamp
        save_dir.mkdir(parents=True, exist_ok=True)

        # 체크포인트 저장
        if is_best:
            save_path = save_dir / "best.pt"
        else:
            save_path = save_dir / f"epoch_{epoch}.pt"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_f1': self.best_f1,  # best_metric 대신 best_f1 사용
        }, save_path)

    def save_predictions(self, predictions, filename="submission.csv"):
        # timestamp 기반 디렉토리 생성
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(self.cfg.output_dir) / "predictions" / timestamp
        save_dir.mkdir(parents=True, exist_ok=True)

        # 예측 결과 저장
        save_path = save_dir / filename
        predictions.to_csv(save_path, index=False)
        print(f"Predictions saved to {save_path}")
