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
import datetime

class ImageDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        """
        Args:
            df: DataFrame or path to csv file
            img_dir: Image directory path
            transform: Albumentations transforms (base_transform, aug_transform)
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
            base_transform, aug_transform = self.transform
            # validation/test는 base_transform만 적용
            img = base_transform(image=img)['image']
        
        return img, row['target']

def get_transforms(img_size: int, augment: str = "basic"):
    """이미지 전처리 및 augmentation 함수 분리"""
    # 1. 기본 전처리 (모든 데이터셋 공통)
    base_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(),
        ToTensorV2(),
    ])
    
    # 2. Augmentation (train only)
    if augment == "none":
        aug_transform = None
    elif augment == "basic":
        aug_transform = A.Compose([
            A.HorizontalFlip(p=0.2),
            A.VerticalFlip(p=0.2),
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
            A.Rotate(limit=30, p=0.2)
        ])
    elif augment == "advanced":
        aug_transform = A.Compose([
            A.OneOf([
                A.RandomRotate90(p=0.5),
                A.Rotate(limit=30, p=0.5),
            ], p=0.5),
            A.OneOf([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
            ], p=0.5),
            # 노이즈 추가 (약하게)
            A.OneOf([
                A.GaussNoise(var_limit=(5.0, 20.0), p=0.5),
                A.GaussianBlur(blur_limit=3, p=0.5),
            ], p=0.3),
            # 색상 변환 (매우 약하게)
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.05,  # ±5%
                    contrast_limit=0.05,    # ±5%
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=5,     # ±5
                    sat_shift_limit=10,    # ±10
                    val_shift_limit=5,     # ±5
                    p=0.5
                ),
            ], p=0.2),
            # 이미지 이동/확대/축소
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=0,
                p=0.3
            ),
        ])
    
    return base_transform, aug_transform

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
        
        # 프로젝트 루트 경로 설정
        self.project_root = Path(__file__).parent.parent
        
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
            
            # HTML 보고서 생성 (augmentation 예시)
            self.create_augmentation_report()
        else:
            self.augment_factors = {cls: 1 for cls in self.class_counts}
        
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
        real_idx = self.indices[idx]
        row = self.df.iloc[real_idx]
        
        # 이미지 로드
        img_path = os.path.join(self.img_dir, row['ID'])
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        if self.transform:
            base_transform, aug_transform = self.transform
            
            if aug_transform:
                # Augmentation 적용 (normalize 이전)
                image = aug_transform(image=image)['image']
            
            # 기본 전처리 적용 (normalize 포함)
            image = base_transform(image=image)['image']
        
        return image, row['target']

    def create_augmentation_report(self):
        """Augmentation 결과 HTML 보고서 생성"""
        from src.embedding import DatasetVisualizer
        
        # 타임스탬프는 한 번만 생성
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = f"outputs/report_augmented/{timestamp}"
        
        visualizer = DatasetVisualizer(
            csv_path=None,
            img_dir=str(self.img_dir),
            output_dir=report_dir,
            timestamp=timestamp  # 기존 타임스탬프 전달
        )
        
        # 클래스별 이미지 경로 설정
        visualizer.class_images = {
            cls: [self.img_dir / row['ID'] 
                 for _, row in self.df[self.df['target'] == cls].iterrows()]
            for cls in self.class_counts.keys()
        }
        
        # 보고서 생성 (augmentation 예시 포함)
        visualizer.create_class_report(
            n_samples=5,  # 각 클래스당 5개 샘플
            seed=42,
            title=f"Augmentation Examples Report - {timestamp}",
            transform=self.transform  # augmentation 함수 전달
        )

def data_prep(data_path: Path, cfg: DictConfig):
    """데이터 준비 및 분할"""
    # 1. 데이터 다운로드 체크
    if not data_path.exists():
        download_and_extract_data(cfg.data.url, data_path)
    
    # 2. CSV 파일 로드
    df = pd.read_csv(data_path / "train.csv")
    
    # 3. 원본 클래스 분포 출력
    class_counts = df['target'].value_counts().sort_index().to_dict()
    print_class_distribution(class_counts, "Original Class Distribution")
    
    # 4. 데이터 분할
    if cfg.data.split_method == "stratified":
        # Stratified Split
        train_df, val_df = train_test_split(
            df, 
            test_size=cfg.data.val_size,
            stratify=df['target'],
            random_state=cfg.seed
        )
        
        # 분할된 데이터 저장
        train_df.to_csv(data_path / "train_fold.csv", index=False)
        val_df.to_csv(data_path / "val_fold.csv", index=False)
        
        # 분할 후 클래스 분포 출력
        print_class_distribution(train_df['target'].value_counts().sort_index().to_dict(), 
                               "Train Set Distribution")
        print_class_distribution(val_df['target'].value_counts().sort_index().to_dict(), 
                               "Validation Set Distribution")
    else:
        # 전체 데이터를 train으로 사용
        df.to_csv(data_path / "train_fold.csv", index=False)
        df.to_csv(data_path / "val_fold.csv", index=False)  # validation도 동일하게 저장

def get_dataloaders(data_path: Path, cfg: DictConfig):
    """데이터로더 생성"""
    train_transform = get_transforms(cfg.train.img_size, augment=cfg.data.augmentation)
    # validation/test는 augmentation 없이 base transform만
    base_transform, _ = get_transforms(cfg.train.img_size, augment="none")
    
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
    
    val_dataset = ImageDataset(
        val_df,
        data_path / "train",
        transform=base_transform  # base transform만 전달
    )
    
    test_dataset = ImageDataset(
        test_df,
        data_path / "test",
        transform=base_transform  # base transform만 전달
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
        print(f"데이터 다운로드 중... (저장 디렉토리: {target_dir})")
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