# CV Context Baseline Ocde
- baseline 과 거의 유사한 설정: config_baseline.yaml 파일 (dummy data 생성, wandb log 만 추가)

## 확장 가능한 옵션들

### 1. Data Augmentation
- `data.augmentation`: 
  - "basic": Resize + Normalize만 적용 (baseline과 동일)
  - "advanced": RandomRotate, RandomBrightnessContrast 등 추가 적용

### 2. Model 선택
- `model.name`:
  - "resnet34": baseline 모델
  - "efficientnet_b0": 대체 모델
  - timm 라이브러리 지원 모델들 사용 가능

### 3. Data Split 방식
- `data.split_method`:
  - "holdout": 단순 분할 (baseline과 동일)
  - "stratified_holdout": 클래스 비율 유지하며 분할
  - "kfold": K-fold 교차 검증
  - "stratified_kfold": 클래스 비율 유지하며 K-fold

### 4. Class Imbalance 처리
- `model.class_balancing.method`:
  - "none": 처리하지 않음 (baseline과 동일)
  - "oversample": 적은 클래스 오버샘플링
    - oversample_strategy: "random" 또는 "smote"
  - "class_weight": 가중치 기반 손실 함수 조정
    - class_weight: "balanced" 또는 "manual"

### 5. Domain Adaptation
- `model.domain_adaptation.method`:
  - "none": 적용하지 않음 (baseline과 동일)
  - "adabn": Adaptive Batch Normalization 적용

### 6. 추가 학습 옵션
- Early Stopping:
  - enabled: true/false
  - patience: 조기 종료 전 대기 에폭 수
  - min_delta: 최소 개선 기준값

- Mixed Precision Training:
  - enabled: true/false
  - dtype: float16/bfloat16

### 7. 로깅
- WandB 로깅:
  - enabled: true/false
  - project_name: 프로젝트명 설정
  - run_name: 실험명 설정

## 사용 예시
configs/config.yaml 파일 설정 변경 후 main.py 실행
configs/config_base.yaml (baseline 설정)이용하려면, main.py hydra 설정시 config name을 config_base로 설정

- wandb logging 활성화 시, wandb 가입 후 실행

### 1. Baseline과 동일한 설정
```yaml
data:
    augmentation: "basic"
    split_method: "holdout"
    test_size: 0.0
model:
    name: "resnet34"
    class_balancing:
        method: "none"
    domain_adaptation:
        method: "none"
train:
    epochs: 1
    batch_size: 32
    img_size: 32
    lr: 1e-3
    early_stopping:
        enabled: false
        patience: 5
    mixed_precision:
        enabled: false
        dtype: float16
logging:
    use_wandb: false
```
### 2. 고급 기능 활용 예시
#### 2-1. Class balancing + Stratified K-fold + Advanced Augmentation + Domain adaptation + EfficientNet + Early stopping + Mixed precision + WandB logging
```yaml
data:
    augmentation: "advanced"
    split_method: "stratified_kfold"
    n_splits: 5
    test_size: 0.2
model:
    name: "efficientnet_b0"
    class_balancing:
        method: "class_weight"
        class_weight: "balanced"
    domain_adaptation:
        method: "adabn"
train:
    epochs: 10
    batch_size: 32
    early_stopping:
        enabled: true
        patience: 5
    mixed_precision:
        enabled: true
        dtype: float16
logging:
    use_wandb: true
    project_name: "cv_contest"
    run_name: "exp_001"
```

### 3. 세부 항목 예시
K-fold + Advanced Augmentation
```yaml

data:
    augmentation: "advanced"
    split_method: "stratified_kfold"
    n_splits: 5
    test_size: 0.2
```

EfficientNet + Class Balancing + domain adaptation
```yaml
model:
    name: "efficientnet_b0"
    class_balancing:
        method: "class_weight"
        class_weight: "balanced"
    domain_adaptation:
        method: "adabn"
```

학습 관련 고급 설정 (추후 기본적인 Hyperparameter 튜닝 포함 예정; num of unfreeze layers 등)
```yaml
train:
    epochs: 10
    batch_size: 32
    early_stopping:
        enabled: true
        patience: 5
    mixed_precision:
        enabled: true
        dtype: float16

```
실험 로깅
```yaml
logging:
    use_wandb: true
    project_name: "cv_contest"
    run_name: "exp_001"
```
### 3. 실행 방법

baseline 설정으로 실행  
```bash
python main.py
```
설정 파일 직접 지정
```bash
python main.py --config-path=. --config-name=config
```
하이드라 오버라이드로 특정 값만 변경
```bash
python main.py model.name=efficientnet_b0 data.augmentation=advanced
```