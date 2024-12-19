import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split, StratifiedKFold
from omegaconf import DictConfig
import subprocess
import shutil

class ImageDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        """
        Args:
            df: DataFrame or path to csv file
            img_dir: Image directory path
            transform: Albumentations transforms
        """
        if isinstance(df, (str, Path)):
            self.df = pd.read_csv(df)
        else:
            self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['ID'])
        img = np.array(Image.open(img_path).convert('RGB'))
        if self.transform:
            img = self.transform(image=img)['image']
        return img, row['target']

def get_transforms(img_size: int, augment: str = "basic"):
    """이미지 전처리 및 augmentation 함수"""
    if augment == "none":
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(),
            ToTensorV2(),
        ])
    elif augment == "basic":
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ])
    elif augment == "advanced":
        return A.Compose([
            A.Resize(img_size, img_size),
            A.OneOf([
                A.RandomRotate90(p=0.5),
                A.Rotate(limit=30, p=0.5),
            ], p=0.5),
            A.OneOf([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
            ], p=0.5),
            A.OneOf([
                A.GaussNoise(p=0.5),
                A.GaussianBlur(p=0.5),
                A.ISONoise(p=0.5),
            ], p=0.3),
            A.OneOf([
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.5),
            ], p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=0,
                p=0.3
            ),
            A.Normalize(),
            ToTensorV2(),
        ])
    else:
        raise ValueError(f"Unknown augmentation: {augment}")

class BalancedImageDataset(Dataset):
    """Class imbalance를 고려한 데이터셋"""
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        
        # 클래스별 샘플 수 계산
        self.class_counts = df['target'].value_counts().to_dict()
        self.max_samples = max(self.class_counts.values())
        
        # 클래스별 augmentation 횟수 계산
        self.augment_factors = {
            cls: max(1, self.max_samples // count)
            for cls, count in self.class_counts.items()
        }
        
        # 인덱스 매핑 생성
        self.indices = []
        for idx, row in df.iterrows():
            cls = row['target']
            self.indices.extend([idx] * self.augment_factors[cls])
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # 실제 데이터프레임 인덱스
        real_idx = self.indices[idx]
        row = self.df.iloc[real_idx]
        
        # 이미지 로드
        img_path = os.path.join(self.img_dir, row['ID'])
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        if self.transform:
            image = self.transform(image=image)['image']
        
        return image, row['target']

def data_prep(data_path: Path, cfg: DictConfig):
    """데이터 준비 및 분할"""
    if not data_path.exists():
        download_and_extract_data(cfg.data.url, data_path)
    
    # CSV 파일 로드 및 컬럼명 확인
    df = pd.read_csv(data_path / "train.csv")
    print("Available columns:", df.columns.tolist())  # 디버깅용
    
    if cfg.data.split_method == "none":
        df.to_csv(data_path / "train_fold.csv", index=False)
        return
    
    # 데이터 분할
    if cfg.data.split_method == "stratified":
        train_df, val_df = train_test_split(
            df, 
            test_size=cfg.data.val_size,
            stratify=df['target'],
            random_state=cfg.seed
        )
    else:
        train_df, val_df = train_test_split(
            df,
            test_size=cfg.data.val_size,
            random_state=cfg.seed
        )
    
    # 저장
    train_df.to_csv(data_path / "train_fold.csv", index=False)
    val_df.to_csv(data_path / "val_fold.csv", index=False)

def get_dataloaders(data_path: Path, cfg: DictConfig):
    """데이터로더 생성"""
    train_transform = get_transforms(cfg.train.img_size, augment=cfg.data.augmentation)
    val_transform = get_transforms(cfg.train.img_size, augment="none")
    
    # 데이터프레임 로드
    train_df = pd.read_csv(data_path / "train_fold.csv")
    val_df = pd.read_csv(data_path / "val_fold.csv")
    test_df = pd.read_csv(data_path / "sample_submission.csv")
    
    # 데이터셋 생성 (class balancing 적용 여부에 따라)
    if cfg.data.get('balance_classes', False):
        train_dataset = BalancedImageDataset(
            train_df,
            data_path / "train",
            transform=train_transform
        )
    else:
        train_dataset = ImageDataset(
            train_df,
            data_path / "train",
            transform=train_transform
        )
    
    val_dataset = ImageDataset(
        val_df,
        data_path / "train",
        transform=val_transform
    )
    
    test_dataset = ImageDataset(
        test_df,
        data_path / "test",
        transform=val_transform
    )
    
    # 데이터로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def download_and_extract_data(url: str, target_dir: Path):
    """데이터 다운로드 및 압축 해제"""
    target_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"데이터 다운로드 중... (저장 위치: {target_dir})")
        temp_dir = target_dir / "temp"
        temp_dir.mkdir(exist_ok=True)
        
        # 다운로드 및 압축 해제
        subprocess.run(['wget', '-q', '--show-progress', url], 
                      cwd=temp_dir, check=True)
        subprocess.run(['tar', '-xf', 'data.tar.gz'], 
                      cwd=temp_dir, check=True)
        
        # 파일 이동
        data_dir = temp_dir / "data"
        if data_dir.exists():
            for item in ['test', 'train', 'train.csv', 'sample_submission.csv']:
                src = data_dir / item
                dst = target_dir / item
                if src.exists():
                    if dst.exists():
                        if dst.is_dir():
                            shutil.rmtree(dst)
                        else:
                            dst.unlink()
                    shutil.move(str(src), str(dst))
        
        # 임시 파일 정리
        shutil.rmtree(temp_dir)
        print("데이터 준비 완료!")
        
    except Exception as e:
        print(f"데이터 다운로드 오류 발생: {str(e)}")
        raise e 