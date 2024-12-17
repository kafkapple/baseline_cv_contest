# model_factory.py
import timm

def get_model(model_name, num_classes=17, pretrained=True):
    if model_name == "resnet34":
        return timm.create_model("resnet34", pretrained=pretrained, num_classes=num_classes)
    elif model_name == "efficientnet_b0":
        return timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")