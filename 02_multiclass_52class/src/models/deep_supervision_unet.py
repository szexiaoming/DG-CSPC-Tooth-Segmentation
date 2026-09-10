"""
Deep Supervision wrapper for U-Net models.

"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AuxSegHead(nn.Module):
    """Lightweight auxiliary segmentation head for deep supervision."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class DeepSupervisionUnet(nn.Module):
    """
    U-Net wrapper with deep supervision from encoder feature pyramid.

    """

    def __init__(
        self,
        base_model: nn.Module,
        num_classes: int = 53,
        aux_scales: Tuple[int, ...] = (1, 2, 3),
        aux_loss_weight: float = 0.3,
    ):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.aux_scales = aux_scales
        self.aux_loss_weight = aux_loss_weight

        # The base model contains encoder, decoder, segmentation_head
        self.encoder = base_model.encoder

        # Determine encoder output channels for each scale
        # We need to do a dummy forward to find channel counts
        encoder_channels = self._get_encoder_channels()

        # Build auxiliary heads
        self.aux_heads = nn.ModuleDict()
        for scale_idx in aux_scales:
            if scale_idx < len(encoder_channels):
                in_ch = encoder_channels[scale_idx]
                self.aux_heads[str(scale_idx)] = AuxSegHead(in_ch, num_classes)

        self._training_mode = True

    def _get_encoder_channels(self) -> List[int]:
        """
        Get output channel counts for each encoder stage.
        
        """
        try:
            # Try to get from encoder's out_channels attribute (smp >= 0.3)
            if hasattr(self.encoder, 'out_channels'):
                return list(self.encoder.out_channels)
        except Exception:
            pass


        try:
            encoder_name = self.encoder.__class__.__name__.lower()
            if 'mobilenet' in encoder_name:
                return [16, 24, 32, 96, 320]  # MobileNetV2
            elif 'efficientnet' in encoder_name:
                return [24, 40, 80, 112, 320]  # EfficientNet-B0
            elif 'resnet' in encoder_name:
                if '18' in encoder_name or '34' in encoder_name:
                    return [64, 64, 128, 256, 512]
                else:  # 50, 101, 152
                    return [64, 256, 512, 1024, 2048]
            elif 'mit' in encoder_name or 'segformer' in encoder_name:
                return [32, 64, 160, 256, 512]  # MiT-B0
        except Exception:
            pass

        # Most conservative fallback
        return [64, 128, 256, 512, 1024]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with deep supervision.

        """
        # Get encoder features
        features = self.encoder(x)  # list of feature maps

        # Main decoder path
        # smp models store decoder and segmentation_head
        # Handle both API versions: decoder(features) and decoder(*features)
        try:
            decoder_output = self.base_model.decoder(*features)
        except TypeError:
            decoder_output = self.base_model.decoder(features)
        main_out = self.base_model.segmentation_head(decoder_output)

        if not self.training or len(self.aux_heads) == 0:
            return main_out

        # Auxiliary outputs from encoder features
        aux_outputs = []
        for scale_idx in self.aux_scales:
            if str(scale_idx) not in self.aux_heads:
                continue
            feat = features[scale_idx]
            aux_logits = self.aux_heads[str(scale_idx)](feat)
            # Upsample to match main output size
            aux_logits = F.interpolate(
                aux_logits,
                size=main_out.shape[2:],
                mode='bilinear',
                align_corners=False,
            )
            aux_outputs.append(aux_logits)

        return {"main": main_out, "aux": aux_outputs}


def create_deep_supervision_model(
    model_name: str = "unet",
    encoder_name: str = "mobilenet_v2",
    in_channels: int = 1,
    num_classes: int = 53,
    pretrained: bool = True,
    pretrained_encoder_path: Optional[str] = None,
    aux_scales: Tuple[int, ...] = (1, 2, 3),
    aux_loss_weight: float = 0.3,
    **kwargs,
) -> DeepSupervisionUnet:
    """
    Create a U-Net model with deep supervision.

    """
    import segmentation_models_pytorch as smp

    model_name_lower = model_name.lower().replace("+", "plus").replace(" ", "")

    if model_name_lower == "unet":
        base = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_channels,
            classes=num_classes,
            **kwargs,
        )
    elif model_name_lower in ("unetpp", "unetplusplus"):
        base = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_channels,
            classes=num_classes,
            **kwargs,
        )
    elif model_name_lower in ("deeplabv3plus", "deeplabv3+"):
        base = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_channels,
            classes=num_classes,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Load SSL-pretrained encoder weights 
    if pretrained_encoder_path:
        from src.models.unet import _load_ssl_encoder_weights
        # The base model's encoder is inside the smp model (not our wrapper)
        base = _load_ssl_encoder_weights(base, pretrained_encoder_path)

    return DeepSupervisionUnet(
        base_model=base,
        num_classes=num_classes,
        aux_scales=aux_scales,
        aux_loss_weight=aux_loss_weight,
    )
