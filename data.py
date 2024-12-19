# data.py
import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from omegaconf import DictConfig
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler, SMOTE
import torch
class ImageDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path).values
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name, target = self.df[idx]
        img = np.array(Image.open(os.path.join(self.img_dir, name)))
        if self.transform:
            img = self.transform(image=img)['image']
        return img, target

def get_transforms(img_size=32):
    trn_transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2(),
    ])
    tst_transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2(),
    ])
    return trn_transform, tst_transform

def data_prep(data_path, cfg: DictConfig):
    """데이터 준비 함수"""
    # baseline 의 경우, split 없이 train 전체 사용
    if cfg.data.split_method == "none":
        # validation split 없이 전체 데이터를 training에 사용
        print(f"데이터 경로: {data_path}")
        train_df = pd.read_csv(data_path / "train.csv")
        train_df.to_csv(data_path / "train_fold.csv", index=False)
        return
    else:
        # 데이터 분할이 필요한 경우
        df = pd.read_csv(data_path / "train.csv")
        data_split(df, data_path, cfg)
#rain 에서  validation 과 test 를 분리하여 사용
def print_distribution(train_df, val_df=None):
    """데이터 분포를 가로(landscape) 테이블 형식으로 출력"""
    print("\n=== 데이터 분포 ===")
    
    # 모든 클래스 목록 준비
    all_classes = sorted(train_df['target'].unique())
    
    # 헤더 출력
    header_line = "데이터셋     총계    |"
    for cls in all_classes:
        header_line += f" 클래스{cls:02d} |"
    print("\n" + "-" * len(header_line))
    print(header_line)
    print("-" * len(header_line))
    
    # 훈련 데이터 분포
    train_dist = train_df['target'].value_counts(normalize=True) * 100
    train_counts = train_df['target'].value_counts()
    
    train_line = f"훈련      {len(train_df):5,d}  |"
    for cls in all_classes:
        count = train_counts.get(cls, 0)
        pct = train_dist.get(cls, 0)
        train_line += f" {count:4d}({pct:4.1f}%) |"
    print(train_line)
    
    # 검증 데이터가 있는 경우
    if val_df is not None:
        val_dist = val_df['target'].value_counts(normalize=True) * 100
        val_counts = val_df['target'].value_counts()
        
        val_line = f"검증      {len(val_df):5,d}  |"
        for cls in all_classes:
            count = val_counts.get(cls, 0)
            pct = val_dist.get(cls, 0)
            val_line += f" {count:4d}({pct:4.1f}%) |"
        print(val_line)
    
    print("-" * len(header_line))
    print()

def data_split(df, data_path, cfg: DictConfig):
    split_method = cfg.data.split_method
    n_splits = cfg.data.get("n_splits", 5)
    fold_index = cfg.data.get("fold_index", 0)
    test_size = cfg.data.get("test_size", 0.2)
    
    if split_method == "holdout":
        # 간단한 holdout split
        train_df, val_df = train_test_split(df, test_size=test_size)
        print_distribution(train_df, val_df)
        
    elif split_method == "stratified_holdout":
        train_df, val_df = train_test_split(df, test_size=test_size, stratify=df['target'])
        print_distribution(train_df, val_df)
        
    else:
        # K-fold or Stratified K-fold
        if split_method == "kfold":
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        elif split_method == "stratified_kfold":
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            raise ValueError("Unsupported split_method")
            
        # fold별로 인덱스 분할
        folds = list(kf.split(df['ID'], df['target']))
        train_idx, val_idx = folds[fold_index]
        
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        
        print(f"\n=== Fold {fold_index + 1}/{n_splits} ===")
        print_distribution(train_df, val_df)

    # 클래스 밸런싱 적용
    if cfg.model.class_balancing.method == "oversample":
        original_train_df = train_df.copy()
        train_df = apply_oversampling(
            train_df, 
            strategy=cfg.model.class_balancing.oversample_strategy
        )
        print("\n=== 오버샘플링 후 분포 ===")
        print_distribution(train_df)

    # 분할 결과 csv 저장
    train_df.to_csv(data_path / "train_fold.csv", index=False)
    val_df.to_csv(data_path / "val_fold.csv", index=False)
    # --- 추가 종료 ---# 
def get_dataloaders(data_path, cfg, batch_size=32, num_workers=0, img_size=32):
    trn_transform, tst_transform = get_transforms(img_size=img_size)
    
    # train_fold.csv와 val_fold.csv를 사용
    train_csv = data_path / "train_fold.csv"
    val_csv = data_path / "val_fold.csv"

    trn_dataset = ImageDataset(
        train_csv,
        data_path / "train",
        transform=trn_transform
    )

    trn_loader = DataLoader(trn_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True)
    
    if cfg.data.split_method == "none":
        val_loader = None
    else:
        val_dataset = ImageDataset(
            val_csv,
            data_path / "train",
            transform=tst_transform
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    tst_dataset = ImageDataset(
        data_path / "sample_submission.csv",
        data_path / "test",
        transform=tst_transform
    )
    tst_loader = DataLoader(tst_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    return trn_loader, val_loader, tst_loader
# 임시 dummy data 생성
def create_dummy_dataset(root="datasets_fin", num_train=100, num_test=20, num_classes=17, img_size=32):
    import os
    from PIL import Image
    import pandas as pd
    os.makedirs(root, exist_ok=True)
    os.makedirs(root / "train", exist_ok=True)
    os.makedirs(root / "test", exist_ok=True)

    # meta.csv
    class_mapping = pd.DataFrame({"target": range(num_classes), "class_name":["class_"+str(i) for i in range(num_classes)]})
    class_mapping.to_csv(root / "meta.csv", index=False)

    # train.csv
    train_data = []
    for i in range(num_train):
        img_name = f"train_{i}.jpg"
        target = np.random.randint(0,num_classes)
        train_data.append([img_name, target])
        # dummy image
        img_array = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
        Image.fromarray(img_array).save(root / "train" / img_name)
    pd.DataFrame(train_data, columns=["ID","target"]).to_csv(root / "train.csv", index=False)

    # sample_submission.csv
    test_data = []
    for i in range(num_test):
        img_name = f"test_{i}.jpg"
        test_data.append([img_name, 0]) # dummy target
        img_array = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
        Image.fromarray(img_array).save(root / "test" / img_name)
    pd.DataFrame(test_data, columns=["ID","target"]).to_csv(root / "sample_submission.csv", index=False)
def get_augmentation(config_name):
    if config_name == "basic": # baseline 코드 기준
        return A.Compose([
            A.Resize(32, 32),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    elif config_name == "advanced":
        return A.Compose([
            A.RandomResizedCrop(32, 32, scale=(0.8, 1.0)),
            A.HorizontalFlip(),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        raise ValueError("Unknown augmentation config.")

def get_class_weights(labels):
    """
    클래스별 가중치 계산
    Args:
        labels: array-like, 클래스 레이블
    Returns:
        torch.Tensor: 각 클래스의 가중치
    """
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    return torch.FloatTensor(class_weights)

def apply_oversampling(df, strategy="random"):
    """
    오버샘플링 적용
    Args:
        df: DataFrame with 'ID' and 'target' columns
        strategy: 'random' or 'smote'
    Returns:
        DataFrame: 오버샘플링이 적용된 데이터프레임
    """
    if strategy == "random":
        sampler = RandomOverSampler(random_state=42)
    elif strategy == "smote":
        sampler = SMOTE(random_state=42)
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    X = df[['ID']]
    y = df['target']
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    
    return pd.DataFrame({'ID': X_resampled['ID'], 'target': y_resampled})