"""
Simple U-Net model using segmentation-models-pytorch.
Provides a factory function to create U-Net variants with different encoders.

"""

import torch
import torch.nn as nn
from typing import Optional


def create_unet(
    encoder_name: str = "resnet34",
    in_channels: int = 1,
    num_classes: int = 1,
    pretrained: bool = True,
    **kwargs,
) -> nn.Module:
    """
    Create a U-Net model using segmentation-models-pytorch.

    """
    import segmentation_models_pytorch as smp

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=in_channels,
        classes=num_classes,
        **kwargs,
    )
    return model


def create_unet_plusplus(
    encoder_name: str = "resnet34",
    in_channels: int = 1,
    num_classes: int = 1,
    pretrained: bool = True,
    **kwargs,
) -> nn.Module:
    """
    Create a U-Net++ model using segmentation-models-pytorch.

    """
    import segmentation_models_pytorch as smp

    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=in_channels,
        classes=num_classes,
        **kwargs,
    )
    return model


def create_model(
    model_name: str,
    encoder_name: str = "resnet34",
    in_channels: int = 1,
    num_classes: int = 1,
    pretrained: bool = True,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create a segmentation model by name.

    """
    import segmentation_models_pytorch as smp

    model_name_lower = model_name.lower().replace("+", "plus").replace(" ", "")

    if model_name_lower in ("unet",):
        return smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_channels,
            classes=num_classes,
            **kwargs,
        )
    elif model_name_lower in ("unetpp", "unetplusplus", "unet++"):
        return smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_channels,
            classes=num_classes,
            **kwargs,
        )
    elif model_name_lower in ("deeplabv3plus", "deeplabv3+", "deeplabv3"):
        return smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_channels,
            classes=num_classes,
            **kwargs,
        )
    elif model_name_lower in ("segformer_b0", "segformerb0"):
        # SegFormer uses MiT (Mix Transformer) encoder
        return smp.Unet(
            encoder_name=encoder_name if encoder_name != "timm-mit_b0" else "mit_b0",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_channels,
            classes=num_classes,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown model name: '{model_name}'. "
            f"Supported: unet, unetpp, deeplabv3plus, segformer_b0"
        )
