# [Computer Vision] Document Type Classification

# I. Overview

## Task: Document Type Classification

| **Category**       | **Type**                    |
|---------------------|-----------------------------|
| **Task Type**       | Recognition → Classification → Multi-class |
| **Data Type**       | Unstructured → Image       |
| **Evaluation Metric** | Macro F1 Score            |

**핵심**: 주어진 이미지 문서 데이터를 **17종 문서 타입**으로 분류하는 Multi-class Classification

## Data

### 📂 Train Data

- **이미지 수**: 1570장
- **클래스**: 17종 (각 클래스당 46~100장)

**구성**:  
- `train.csv`: 이미지 ID와 클래스(Target) 매핑  
- `train/`: 실제 학습 이미지 폴더  

```csv
ID,target
image1.jpg,0
image2.jpg,1
image3.jpg,2
...
```


### 📂 Test Data
- **이미지 수**: 3140장 (Train보다 많음)
- **특징**: Augmentation을 통해 현실 세계의 노이즈가 추가됨

**구성**:
- sample_submission.csv: 예측값을 채워 넣을 더미 파일
- test/: 테스트 이미지 폴더

```csv    
ID,target
test_image1.jpg,0
test_image2.jpg,1
...
```

데이터 요약
- Train: 클래스당 46~100개의 이미지, 비교적 깨끗한 데이터
- Test: Augmentation 및 Noise 추가, Train보다 데이터 양이 많음
- Target: class_name과 index는 meta.csv서 확인 가능

## Evaluation Metric: Macro F1 Score
- F1 Score: Precision과 Recall의 **조화 평균**  
- **Macro F1 Score**  
  각 클래스별 F1 Score를 **평균**하여 계산 (class imbalance 에 예민)
- **Micro F1 Score**  
  모든 클래스의 Precision과 Recall을 **합산**하여 계산  

> **이 대회에서는 `Macro F1 Score`를 사용**

---

# II. Project Setup

## 1. Server Setup (Optional)

### 자동 설정: setup_server.sh 
```plaintext
- Git 설정 (사용자 이름, 이메일, PAT)
- Conda 초기화
- 기본 Linux 패키지 설치 (JDK, vim 등)
```
### Method 
- setup_server.sh 파일 서버에 복사
- 권한 설정 및 실행
```bash
chmod +x setup_server.sh
bash setup_server.sh
```
- Github 사용자 이름, 이메일, PAT 입력
- Github repo, upstream 설정 (Optional)




## 1. Installation
```bash
# 1. Repository Clone
git clone https://github.com/kafkapple/baseline_cv_contest.git

cd baseline_cv_contest

# 2. Environment Setup
conda env create -f environment.yml


```

## 2. Weights & Biases Setup
```yaml
# configs/config.yaml
logger:
  use_wandb: true
  entity: "ailab_upstage_fastcampus"
  project_name: "ailab_contest_cv_team_1"
```

# III. Project Structure

## 1. Directory Structure
```
📦 baseline_cv_contest
├── 📜 main.py              # 실행 스크립트
├── 📜 setup_server.sh      # 서버 환경 설정
├── 📂 src/
│   ├── 📜 data.py         # 데이터셋 및 augmentation
│   ├── 📜 trainer.py      # 학습 관리
│   ├── 📜 logger.py       # W&B 로깅
│   ├── 📜 embedding.py    # 데이터 시각화
│   └── 📜 models.py       # 모델 아키텍처
├── 📂 configs/
│   └── 📜 config.yaml     # 메인 설정 파일
└── 📂 outputs/            # 실험 결과물
    ├── 📂 checkpoints/    # 모델 저장
    ├── 📂 predictions/    # 예측 결과
    └── 📂 report_augmented/ # Augmentation 리포트
```

## 2. Experiment Management

### 2.1 실험 결과 구조
```
outputs/
├── checkpoints/                    # 모델 체크포인트
│   └── {timestamp}/
│       ├── best.pt                # 최고 성능 모델
│       └── epoch_{N}.pt           # 에폭별 체크포인트
│
├── predictions/                    # 예측 결과
│   └── {timestamp}/
│       ├── submission.csv         # 테스트 예측
│       └── val_predictions.csv    # 검증 예측
│
├── report_augmented/              # Augmentation 시각화
│   └── {timestamp}/
│       ├── images/               # 원본/증강 이미지
│       └── class_report.html     # 분석 리포트
│
└── logs/                          # 실험 로그
    └── {timestamp}/
        ├── config.json           # 실험 설정
        └── metrics.json          # 성능 지표
```

### 2.2 실험 ID 관리
- **형식**: `YYYYMMDD_HHMMSS`
- **사용**: 모든 실험 결과물이 동일 timestamp로 저장
- **W&B Run Name**: `{timestamp}_{model}_{augmentation}`

# IV. Configuration Guide

## 1. 주요 설정 옵션
```yaml
data:
  # 1. Augmentation
  augmentation: "advanced"  # [none, basic, advanced]
  
  # 2. 데이터 분할
  split_method: "stratified"  # [stratified, kfold]
  val_size: 0.2
  
  # 3. 클래스 불균형 처리
  balance_strategy: "augmentation"  # [none, augmentation, weighted_sampler]
  aug_strategy: "target"  # [median, max, target]
  target_count: 1000

model:
  # 1. 모델 선택
  name: "efficientnet_b0"  # [resnet18-101, efficientnet_b0-b7, vit_*]
  
  # 2. 학습 설정
  pretrained: true
  regularization:
    dropout: 0.0
    label_smoothing: 0.2

train:
  epochs: 3
  batch_size: 64
  img_size: 224
  lr: 1e-3
  early_stopping:
    enabled: true
    patience: 5
```

# V. Usage

## 1. Training
```bash
# 기본 실행
python main.py

# 설정 오버라이드
python main.py model.name=efficientnet_b0 data.augmentation=advanced
```

## 2. Monitoring
- **W&B Dashboard**: 실시간 학습 현황
  - Parameters: 모델 설정, 데이터 설정
  - Metrics: 손실, 정확도, F1 Score
  - Artifacts: 모델 체크포인트

- **Local Reports**
  - HTML 리포트: Augmentation 결과 시각화
  - Predictions: CSV 형식의 예측 결과
    - 최고 성능 모델의 F1 Score 기반
  - Logs: 실험 설정 및 메트릭

## 3. Git Management
```gitignore
# 버전 관리 제외 항목
*.pt          # 모델 체크포인트
wandb/        # W&B 로그
outputs/      # 실험 결과물
data/         # 데이터셋
logs/         # 로그 파일
```
