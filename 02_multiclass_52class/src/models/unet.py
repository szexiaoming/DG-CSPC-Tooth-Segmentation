"""
Simple U-Net model using segmentation-models-pytorch.

"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional


def _load_ssl_encoder_weights(
    model: nn.Module,
    pretrained_encoder_path: str,
    device: str = "cpu",
    verbose: bool = True,
) -> nn.Module:
    
    """
    Load SSL-pretrained (MoCo/SimCLR/BYOL) encoder backbone weights into an smp model.

    """
    ckpt = torch.load(pretrained_encoder_path, map_location=device)

    # Support two checkpoint formats:
    #   1. {"backbone": OrderedDict, "config": {...}}  ← pretrain_moco.py output
    #   2. Raw OrderedDict (backbone state_dict directly)
    if isinstance(ckpt, dict) and "backbone" in ckpt:
        backbone_state = ckpt["backbone"]
        cfg = ckpt.get("config", {})
        if verbose:
            print(f"[SSL] Loading {cfg.get('method', 'SSL')} pretrained encoder: "
                  f"{cfg.get('backbone', '?')}, {cfg.get('pretrain_epochs', '?')} epochs")
    else:
        backbone_state = ckpt
        if verbose:
            print("[SSL] Loading raw encoder state_dict (format auto-detected)")

    # Load into smp's encoder.
    encoder = model.encoder
    encoder_target = encoder

    # Detect smp TimmEncoder wrapper
    if hasattr(encoder, 'model') and hasattr(encoder.model, 'load_state_dict'):
        # Check whether keys match the wrapper or the inner model
        first_key = next(iter(backbone_state.keys()), "")
        if first_key and first_key not in dict(encoder.state_dict()):
            encoder_target = encoder.model
            if verbose:
                print(f"  → Loading into encoder.model (smp TimmEncoder wrapper)")

    # Load weights (handle PyTorch version differences)
    result = encoder_target.load_state_dict(backbone_state, strict=False)
    if result is None:
        # PyTorch >= 2.4 returns None with strict=False — verify keys manually
        loaded_keys = set(backbone_state.keys())
        target_keys = set(encoder_target.state_dict().keys())
        missing = sorted(target_keys - loaded_keys)
        unexpected = sorted(loaded_keys - target_keys)
    else:
        missing, unexpected = result

    # Filter out projection-head / fc keys (not part of encoder backbone)
    proj_patterns = ("projection.", "fc.", "classifier.", "head.")
    real_missing = [
        k for k in missing
        if not any(p in k for p in proj_patterns)
    ]
    safe_unexpected = [
        k for k in unexpected
        if any(p in k for p in proj_patterns)
    ]

    if verbose:
        if real_missing:
            print(f" {len(real_missing)} keys not matched: "
                  f"{real_missing[:3]}{'...' if len(real_missing) > 3 else ''}")
        elif safe_unexpected:
            print(f" {len(safe_unexpected)} projection-head keys skipped")
        else:
            match_pct = (len(backbone_state) - len(real_missing)) / max(len(backbone_state), 1) * 100
            print(f" {match_pct:.0f}% keys matched ({len(backbone_state)} total)")

    return model


def create_unet(
    encoder_name: str = "resnet34",
    in_channels: int = 1,
    num_classes: int = 1,
    pretrained: bool = True,
    pretrained_encoder_path: Optional[str] = None,
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

    if pretrained_encoder_path:
        model = _load_ssl_encoder_weights(model, pretrained_encoder_path)

    return model


def create_unet_plusplus(
    encoder_name: str = "resnet34",
    in_channels: int = 1,
    num_classes: int = 1,
    pretrained: bool = True,
    pretrained_encoder_path: Optional[str] = None,
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

    if pretrained_encoder_path:
        model = _load_ssl_encoder_weights(model, pretrained_encoder_path)

    return model


def create_model(
    model_name: str,
    encoder_name: str = "resnet34",
    in_channels: int = 1,
    num_classes: int = 1,
    pretrained: bool = True,
    pretrained_encoder_path: Optional[str] = None,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create a segmentation model by name.

    """
    import segmentation_models_pytorch as smp

    model_name_lower = model_name.lower().replace("+", "plus").replace(" ", "")

    if model_name_lower in ("unet",):
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_channels,
            classes=num_classes,
            **kwargs,
        )
    elif model_name_lower in ("unetpp", "unetplusplus", "unet++"):
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_channels,
            classes=num_classes,
            **kwargs,
        )
    elif model_name_lower in ("deeplabv3plus", "deeplabv3+", "deeplabv3"):
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_channels,
            classes=num_classes,
            **kwargs,
        )
    elif model_name_lower in ("segformer_b0", "segformerb0"):
        # SegFormer uses MiT (Mix Transformer) encoder
        model = smp.Unet(
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

    if pretrained_encoder_path:
        model = _load_ssl_encoder_weights(model, pretrained_encoder_path)

    return model
