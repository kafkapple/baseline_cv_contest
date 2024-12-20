# model_factory.py
import timm
import torch
import torch.nn as nn
import torch.nn.init as init
from omegaconf import DictConfig, OmegaConf

def initialize_weights(model: nn.Module, method: str = "kaiming", **kwargs):
    """가중치 초기화"""
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if method == "xavier_uniform":
                init.xavier_uniform_(m.weight, gain=kwargs.get('gain', 1.0))
            elif method == "xavier_normal":
                init.xavier_normal_(m.weight, gain=kwargs.get('gain', 1.0))
            elif method == "kaiming_uniform":
                init.kaiming_uniform_(m.weight, 
                                    mode=kwargs.get('mode', 'fan_in'),
                                    nonlinearity=kwargs.get('nonlinearity', 'relu'))
            elif method == "kaiming_normal":
                init.kaiming_normal_(m.weight, 
                                   mode=kwargs.get('mode', 'fan_in'),
                                   nonlinearity=kwargs.get('nonlinearity', 'relu'))
            
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

class AdaptiveModel(nn.Module):
    def __init__(self, base_model: nn.Module, num_classes: int, cfg: DictConfig):
        super().__init__()
        self.method = cfg.model.domain_adaptation.method
        
        # Feature extractor (기존 모델의 마지막 FC layer 제외)
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        feature_dim = base_model.fc.in_features
        
        # Dropout 추가
        self.dropout = nn.Dropout(p=cfg.model.regularization.dropout)
        
        # Task classifier
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        # Domain adaptation 관련 layers
        if self.method == "tta":
            self.rot_classifier = nn.Linear(feature_dim, 4)
        elif self.method == "dann":
            self.domain_classifier = nn.Sequential(
                nn.Linear(feature_dim, 1024),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1024, 2)  # 2 domains (source/target)
            )
        
        # Weight initialization (pretrained가 아닐 경우에만)
        if not cfg.model.pretrained and cfg.model.init_method != "none":
            initialize_weights(self, 
                            method=cfg.model.init_method,
                            **cfg.model.init_params)
    
    def forward(self, x, mode='main', alpha=None):
        features = self.features(x)
        features = torch.flatten(features, 1)
        features = self.dropout(features)
        
        if mode == 'main':
            if self.method == "dann" and alpha is not None:
                # DANN: return both class and domain predictions
                class_output = self.classifier(features)
                domain_output = self.domain_classifier(features)
                return class_output, domain_output
            else:
                # Normal forward pass
                return self.classifier(features)
        elif mode == 'rotation':
            return self.rot_classifier(features)

def get_model(model_name: str, num_classes: int, pretrained: bool = True, cfg: DictConfig = None):
    base_model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    
    # cfg가 None인 경우 기본 설정으로 생성
    if cfg is None:
        cfg = OmegaConf.create({
            'model': {
                'pretrained': pretrained,
                'domain_adaptation': {'method': 'none'},
                'regularization': {
                    'dropout': 0.0,
                    'weight_decay': 0.0,
                    'label_smoothing': 0.0
                },
                'init_method': 'none',
                'init_params': {
                    'gain': 1.0,
                    'mode': 'fan_in',
                    'nonlinearity': 'relu'
                }
            }
        })
    
    # AdaptiveModel로 감싸서 반환
    return AdaptiveModel(base_model, num_classes, cfg)