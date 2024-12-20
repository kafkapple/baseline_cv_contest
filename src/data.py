import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from omegaconf import DictConfig
import subprocess
import shutil
from torch.utils.data import WeightedRandomSampler

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

def print_class_distribution(counts: dict, title: str = "Class Distribution"):
    """클래스 분포를 테이블로 출력"""
    # 데이터프레임으로 변환하여 깔끔하게 출력
    df = pd.DataFrame({
        'Class': list(counts.keys()),
        'Count': list(counts.values()),
        'Ratio(%)': [count/sum(counts.values())*100 for count in counts.values()]
    })
    
    print(f"\n=== {title} ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    print(f"\nTotal Samples: {sum(counts.values()):,}")
    print("=" * 40)
    print()

def calculate_augment_factors(class_counts: dict, strategy: str = "median", target_count: int = None):
    """augmentation 배수 계산
    Args:
        strategy: "median", "max", "target" 중 하나
        target_count: strategy가 "target"일 때 목표 샘플 수
    """
    if strategy == "median":
        target = np.median(list(class_counts.values()))
    elif strategy == "max":
        target = max(class_counts.values())
    elif strategy == "target":
        target = target_count if target_count else np.median(list(class_counts.values()))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return {
        cls: max(1, int(np.ceil(target / count)))
        for cls, count in class_counts.items()
    }

class BalancedImageDataset(Dataset):
    """Class imbalance를 고려한 데이터셋"""
    def __init__(self, df, img_dir, transform=None, cfg=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        
        # 클래스별 샘플 수 계산
        self.class_counts = df['target'].value_counts().to_dict()
        self.max_samples = max(self.class_counts.values())
        self.min_samples = min(self.class_counts.values())
        
        # 클래스별 augmentation 횟수 계산
        if cfg and cfg.data.balance_strategy == "augmentation":
            self.augment_factors = calculate_augment_factors(
                self.class_counts,
                strategy=cfg.data.get('aug_strategy', 'median'),
                target_count=cfg.data.get('target_count', None)
            )
        else:
            self.augment_factors = {cls: 1 for cls in self.class_counts}  # 기본 augmentation만
        
        # 인덱스 매핑 생성
        self.indices = []
        for idx, row in df.iterrows():
            cls = row['target']
            self.indices.extend([idx] * self.augment_factors[cls])
        
        # Augmentation 후 클래스 분포 출력
        augmented_counts = {
            cls: count * self.augment_factors[cls] 
            for cls, count in self.class_counts.items()
        }
        print_class_distribution(augmented_counts, "Augmented Class Distribution")
    
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
    
    # CSV 파일 로드
    df = pd.read_csv(data_path / "train.csv")
    
    # 클래스 분포 출력
    class_counts = df['target'].value_counts().sort_index().to_dict()
    print_class_distribution(class_counts, "Original Class Distribution")
    
    if cfg.data.split_method == "none":
        df.to_csv(data_path / "train_fold.csv", index=False)
        return
    
    # 데이터 분할 및식에 따라 처리
    if cfg.data.split_method == "stratified_kfold":
        # Stratified K-Fold Cross Validation
        skf = StratifiedKFold(
            n_splits=cfg.data.n_splits,
            shuffle=True,
            random_state=cfg.seed
        )
        
        # 현재 fold 인덱스에 해당하는 split 찾기
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, df['target'])):
            if fold_idx == cfg.data.fold_index:
                train_df = df.iloc[train_idx]
                val_df = df.iloc[val_idx]
                break
                
        print(f"\nUsing fold {cfg.data.fold_index + 1}/{cfg.data.n_splits}")
        
    elif cfg.data.split_method == "kfold":
        # Regular K-Fold Cross Validation
        kf = KFold(
            n_splits=cfg.data.n_splits,
            shuffle=True,
            random_state=cfg.seed
        )
        
        # 현재 fold 인덱스에 해당하는 split 찾기
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(df)):
            if fold_idx == cfg.data.fold_index:
                train_df = df.iloc[train_idx]
                val_df = df.iloc[val_idx]
                break
                
        print(f"\nUsing fold {cfg.data.fold_index + 1}/{cfg.data.n_splits}")
        
    elif cfg.data.split_method == "stratified":
        # Stratified Split
        train_df, val_df = train_test_split(
            df, 
            test_size=cfg.data.val_size,
            stratify=df['target'],
            random_state=cfg.seed
        )
    
    elif cfg.data.split_method == "holdout":
        # Regular Split
        train_df, val_df = train_test_split(
            df,
            test_size=cfg.data.val_size,
            random_state=cfg.seed
        )
    
    else:
        raise ValueError(f"Unknown split method: {cfg.data.split_method}")
    
    # 분할된 데이터 저장
    train_df.to_csv(data_path / "train_fold.csv", index=False)
    val_df.to_csv(data_path / "val_fold.csv", index=False)
    
    # 분할 후 클래스 분포 출력
    print_class_distribution(train_df['target'].value_counts().sort_index().to_dict(), 
                           "Train Set Distribution")
    print_class_distribution(val_df['target'].value_counts().sort_index().to_dict(), 
                           "Validation Set Distribution")

def get_dataloaders(data_path: Path, cfg: DictConfig):
    """데이터로더 생성"""
    train_transform = get_transforms(cfg.train.img_size, augment=cfg.data.augmentation)
    val_transform = get_transforms(cfg.train.img_size, augment="none")
    
    # 데이터프레임 로드
    train_df = pd.read_csv(data_path / "train_fold.csv")
    val_df = pd.read_csv(data_path / "val_fold.csv")
    test_df = pd.read_csv(data_path / "sample_submission.csv")
    
    # 데이터셋 생성
    if cfg.data.balance_strategy == "augmentation":
        train_dataset = BalancedImageDataset(
            train_df,
            data_path / "train",
            transform=train_transform,
            cfg=cfg
        )
    else:
        train_dataset = ImageDataset(
            train_df,
            data_path / "train",
            transform=train_transform
        )
    
    # WeightedRandomSampler 설정
    if cfg.data.balance_strategy == "weighted_sampler":
        class_counts = train_df['target'].value_counts().to_dict()
        weights = [1.0 / class_counts[target] for target in train_df['target']]
        sampler = WeightedRandomSampler(weights, len(weights))
    else:
        sampler = None
    
    # 데이터로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg.train.num_workers,
        pin_memory=True
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
    target_dir = Path(target_dir).resolve()
    print(f"현재 작업 디렉토리: {os.getcwd()}")
    print(f"대상 디렉토리 (절대경로): {target_dir}")
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"데이터 다운��드 중... (저장 위치: {target_dir})")
        temp_dir = target_dir / "temp"
        temp_dir.mkdir(exist_ok=True)
        
        # 다운로드 및 압축 해제 (절대 경로 사용)
        subprocess.run(['wget', '-q', '--show-progress', url], 
                      cwd=str(temp_dir.absolute()), check=True)
        subprocess.run(['tar', '-xf', 'data.tar.gz'], 
                      cwd=str(temp_dir.absolute()), check=True)
        
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