# model_factory.py
import timm
import torch
import torch.nn as nn

class AdaptiveModel(nn.Module):
    def __init__(self, base_model: nn.Module, num_classes: int, method: str = "none"):
        super().__init__()
        self.method = method
        
        # Feature extractor (기존 모델의 마지막 FC layer 제외)
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        feature_dim = base_model.fc.in_features
        
        # Task classifier
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        # Rotation classifier for test-time adaptation
        if method == "tta":
            self.rot_classifier = nn.Linear(feature_dim, 4)  # 4가지 회전 각도 예측
    
    def forward(self, x, mode='main'):
        features = self.features(x)
        features = torch.flatten(features, 1)
        
        if mode == 'main':
            return self.classifier(features)
        elif mode == 'rotation':
            return self.rot_classifier(features)

def get_model(model_name: str, num_classes: int, pretrained: bool = True, domain_adaptation: str = "none"):
    base_model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    
    # 항상 AdaptiveModel로 감싸서 반환 (domain adaptation 여부와 관계없이)
    return AdaptiveModel(base_model, num_classes, method=domain_adaptation)