"""
TransferBench
==============

Model Factory

Supported Models

- ResNet18
- EfficientNet-B0
- Vision Transformer (ViT-B16)

Author:
TransferBench
"""

from torchvision.models import (
    resnet18,
    efficientnet_b0,
    vit_b_16,
    ResNet18_Weights,
    EfficientNet_B0_Weights,
    ViT_B_16_Weights,
)

import torch.nn as nn


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def freeze_model(model):
    """
    Freeze every layer.
    """

    for p in model.parameters():
        p.requires_grad = False

    return model


def unfreeze_layer(module):
    """
    Unfreeze selected module.
    """

    for p in module.parameters():
        p.requires_grad = True


# --------------------------------------------------
# ResNet18
# --------------------------------------------------

def build_resnet18(
    num_classes,
    strategy="freeze"
):

    model = resnet18(
        weights=ResNet18_Weights.DEFAULT
    )

    freeze_model(model)

    model.fc = nn.Sequential(

        nn.Dropout(0.30),

        nn.Linear(
            model.fc.in_features,
            num_classes,
        ),
    )

    unfreeze_layer(model.fc)

    if strategy == "finetune":

        unfreeze_layer(model.layer4)

    return model


# --------------------------------------------------
# EfficientNet
# --------------------------------------------------

def build_efficientnet_b0(
    num_classes,
    strategy="freeze"
):

    model = efficientnet_b0(

        weights=EfficientNet_B0_Weights.DEFAULT

    )

    freeze_model(model)

    model.classifier = nn.Sequential(

        nn.Dropout(0.30),

        nn.Linear(

            model.classifier[1].in_features,

            num_classes,

        ),

    )

    unfreeze_layer(model.classifier)

    if strategy == "finetune":

        unfreeze_layer(model.features[-1])

    return model


# --------------------------------------------------
# Vision Transformer
# --------------------------------------------------

def build_vit_b16(
    num_classes,
    strategy="freeze"
):

    model = vit_b_16(

        weights=ViT_B_16_Weights.DEFAULT

    )

    freeze_model(model)

    model.heads.head = nn.Linear(

        model.heads.head.in_features,

        num_classes,

    )

    unfreeze_layer(model.heads)

    if strategy == "finetune":

        unfreeze_layer(model.encoder.layers[-1])

    return model


# --------------------------------------------------
# Factory
# --------------------------------------------------

def build_model(
    model_name,
    num_classes,
    strategy="freeze",
):

    model_name = model_name.lower()

    if model_name == "resnet18":

        return build_resnet18(
            num_classes,
            strategy,
        )

    elif model_name == "efficientnet":

        return build_efficientnet_b0(
            num_classes,
            strategy,
        )

    elif model_name == "vit":

        return build_vit_b16(
            num_classes,
            strategy,
        )

    else:

        raise ValueError(

            f"Unknown model : {model_name}"

        )


# --------------------------------------------------
# Information
# --------------------------------------------------

def count_parameters(model):

    total = sum(

        p.numel()

        for p in model.parameters()

    )

    trainable = sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad

    )

    frozen = total - trainable

    return {

        "total": total,

        "trainable": trainable,

        "frozen": frozen,

    }