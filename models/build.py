"""Factory de modelos preentrenados para clasificación multietiqueta."""
import torch.nn as nn
from torchvision import models


def _replace_classifier(model, num_classes):
    """Sustituye la última capa por una nueva con num_classes salidas."""
    if hasattr(model, "fc"):  # ResNet
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif hasattr(model, "classifier"):  # DenseNet
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    else:
        raise ValueError("Arquitectura no soportada")
    return model


def _apply_freeze_strategy(model, strategy):
    """Congela capas según la estrategia de fine-tuning."""
    if strategy == "full":
        return model
    for param in model.parameters():
        param.requires_grad = False
    # Descongelar siempre el clasificador
    for param in model.fc.parameters() if hasattr(model, "fc") else model.classifier.parameters():
        param.requires_grad = True
    if strategy == "partial":
        # Descongelar el último bloque (ejemplo genérico)
        if hasattr(model, "layer4"):
            for param in model.layer4.parameters():
                param.requires_grad = True
        elif hasattr(model, "features"):
            for param in model.features[-1].parameters():
                param.requires_grad = True
    return model


def build_model(name, num_classes, pretrained=True, freeze_strategy="full"):
    """Construye un modelo preentrenado y aplica la estrategia de congelación."""
    if name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
    elif name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
    else:
        raise ValueError(f"Modelo no soportado: {name}")

    model = _replace_classifier(model, num_classes)
    model = _apply_freeze_strategy(model, freeze_strategy)
    return model
