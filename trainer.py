# trainer.py
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from typing import Dict, Any, Optional

class Trainer:
    def __init__(self, model, device, optimizer, loss_fn, logger=None, class_weights=None, cfg=None):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.loss_fn = loss_fn if class_weights is None else nn.CrossEntropyLoss(weight=class_weights.to(device))
        self.logger = logger
        self.cfg = cfg
        
    def update_bn_stats(self, loader):
        """AdaBN: 테스트 데이터로 BatchNorm 통계 업데이트"""
        self.model.train()  # BN 통계를 업데이트하기 위해 train 모드로 설정
        with torch.no_grad():  # 그래디언트는 계산하지 않음
            for images, _ in tqdm(loader, desc="Updating BN statistics"):
                images = images.to(self.device)
                _ = self.model(images)  # forward pass만 수행하여 BN 통계 업데이트

    def inference(self, loader):
        """추론 시 Domain Adaptation 적용"""
        if self.cfg and self.cfg.model.domain_adaptation.method == "adabn":
            print("Applying AdaBN...")
            # 테스트 데이터로 BN 통계 업데이트
            self.update_bn_stats(loader)
        
        # 일반적인 추론 수행
        self.model.eval()
        preds_list = []
        with torch.no_grad():
            for image, _ in tqdm(loader, desc="Inference"):
                image = image.to(self.device)
                preds = self.model(image)
                preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())
        
        return preds_list

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

    def train_one_epoch(self, loader):
        self.model.train()
        train_loss = 0
        preds_list = []
        targets_list = []

        for image, targets in tqdm(loader, desc="Training"):
            image, targets = image.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            preds = self.model(image)
            loss = self.loss_fn(preds, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())
            targets_list.extend(targets.detach().cpu().numpy())

        train_loss /= len(loader)
        preds_array = np.array(preds_list)
        targets_array = np.array(targets_list)
        
        return self._calculate_metrics(preds_array, targets_array, train_loss, "train")

    def evaluate(self, loader):
        self.model.eval()
        val_loss = 0
        preds_list = []
        targets_list = []

        with torch.no_grad():
            for image, targets in tqdm(loader, desc="Validation"):
                image, targets = image.to(self.device), targets.to(self.device)
                preds = self.model(image)
                loss = self.loss_fn(preds, targets)

                val_loss += loss.item()
                preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())
                targets_list.extend(targets.detach().cpu().numpy())

        val_loss /= len(loader)
        preds_array = np.array(preds_list)
        targets_array = np.array(targets_list)

        return self._calculate_metrics(preds_array, targets_array, val_loss, "val")

    def train(self, train_loader, val_loader, epochs):
        for epoch in range(epochs):
            # 학습
            train_metrics = self.train_one_epoch(train_loader)
            
            # 검증
            val_metrics = self.evaluate(val_loader) if val_loader is not None else {}
            
            # 전역 메트릭 추가
            metrics = {
                **train_metrics,
                **val_metrics,
                "global/epoch": epoch,
                "global/learning_rate": self.optimizer.param_groups[0]['lr']
            }
            
            # 로깅
            if self.logger is not None:
                self.logger.log(metrics)
            
            # 결과 출력
            self._print_metrics(epoch, epochs, train_metrics, val_metrics)
    
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
        
        # 클래스별 F1 출력
        # if self.cfg.model.num_classes > 1:
        #     print("\nClass-wise F1 scores:")
        #     print("Train:", end=" ")
        #     for i in range(self.cfg.model.num_classes):
        #         print(f"Class {i}: {train_metrics[f'train/f1_class_{i}']:.4f}", end="  ")
            
        #     if val_metrics:
        #         print("\nVal  :", end=" ")
        #         for i in range(self.cfg.model.num_classes):
        #             print(f"Class {i}: {val_metrics[f'val/f1_class_{i}']:.4f}", end="  ")
            
            print()
        
