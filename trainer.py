# trainer.py
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

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
        train_acc = accuracy_score(targets_list, preds_list)
        train_f1 = f1_score(targets_list, preds_list, average='macro')
        
        metrics = {
            "train_loss": train_loss, 
            "train_acc": train_acc, 
            "train_f1": train_f1
        }
        
        # wandb logging을 위한 클래스별 F1 score (출력하지는 않음)
        if self.logger is not None:
            class_f1 = f1_score(targets_list, preds_list, average=None)
            for i, f1 in enumerate(class_f1):
                metrics[f"train_f1_class_{i}"] = f1
            
        return metrics

    def evaluate(self, loader, prefix="val"):
        """Evaluate model on validation/test data"""
        self.model.eval()
        total_loss = 0
        preds_list = []
        targets_list = []

        with torch.no_grad():
            for image, targets in tqdm(loader, desc=f"{prefix.capitalize()} Evaluation"):
                image, targets = image.to(self.device), targets.to(self.device)
                preds = self.model(image)
                loss = self.loss_fn(preds, targets)
                
                total_loss += loss.item()
                preds_list.extend(preds.argmax(dim=1).cpu().numpy())
                targets_list.extend(targets.cpu().numpy())

        avg_loss = total_loss / len(loader)
        acc = accuracy_score(targets_list, preds_list)
        f1 = f1_score(targets_list, preds_list, average='macro')
        
        return {
            f"{prefix}_loss": avg_loss,
            f"{prefix}_acc": acc,
            f"{prefix}_f1": f1
        }

    def train(self, train_loader, val_loader, epochs):
        for epoch in range(epochs):
            train_metrics = self.train_one_epoch(train_loader)
            
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                # validation 관련 로깅
                metrics = {**train_metrics, **val_metrics, "epoch": epoch}
                if self.logger is not None:
                    self.logger.log(metrics)
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"Train - Loss: {train_metrics['train_loss']:.4f}, F1: {train_metrics['train_f1']:.4f}")
                print(f"Val   - Loss: {val_metrics['val_loss']:.4f}, F1: {val_metrics['val_f1']:.4f}")
            else:
                # validation 없이 training 메트릭만 로깅
                if self.logger is not None:
                    self.logger.log(train_metrics)
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"Train - Loss: {train_metrics['train_loss']:.4f}, F1: {train_metrics['train_f1']:.4f}")
            
            print("-" * 50)

    def inference(self, loader):
        self.model.eval()
        preds_list = []
        with torch.no_grad():
            for image, _ in tqdm(loader, desc="Inference"):
                image = image.to(self.device)
                preds = self.model(image)
                preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())
        return preds_list
        
