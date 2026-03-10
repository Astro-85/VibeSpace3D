"""Week-1 shared trainer scaffolding for VibeSpace3D.

This module defines a lightweight training orchestration layer that can be
extended by Track A (multi-view diffusion) and Track B (triplane decoding).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from src.vibespace_model import VibeSpaceModel, train_vibe_space


@dataclass
class VibeSpace3DTrainConfig:
    steps: int = 1000
    vibe_dim: int = 64
    ipadapter_version: str = "sd15"
    min_correspondence_confidence: float = 0.2
    max_views: Optional[int] = None


def prepare_multiview_token_batch(
    features: torch.Tensor,
    max_views: Optional[int] = None,
) -> torch.Tensor:
    """Flatten multi-view features into a batch format compatible with VibeSpaceModel.

    Args:
        features: (B, V, T, D) or (V, T, D)
    Returns:
        Tensor: (B*V, T, D)
    """
    if features.dim() == 3:
        features = features.unsqueeze(0)
    if features.dim() != 4:
        raise ValueError(f"Expected (B,V,T,D) or (V,T,D), got {tuple(features.shape)}")

    if max_views is not None:
        features = features[:, :max_views]

    b, v, t, d = features.shape
    return features.reshape(b * v, t, d)


def train_vibespace3d_from_features(
    model: VibeSpaceModel,
    config,
    positive_dino_features: torch.Tensor,
    target_clip_features: torch.Tensor,
    negative_dino_features: Optional[torch.Tensor] = None,
    devices=None,
) -> Dict[str, torch.Tensor | object]:
    """Run shared training entrypoint from pre-extracted multiview features."""
    if devices is None:
        devices = [0]

    train_input = prepare_multiview_token_batch(positive_dino_features)
    train_target = prepare_multiview_token_batch(target_clip_features)
    train_negative = (
        None
        if negative_dino_features is None
        else prepare_multiview_token_batch(negative_dino_features)
    )

    trainer = train_vibe_space(
        model=model,
        config=config,
        input_features=train_input,
        target_features=train_target,
        negative_features=train_negative,
        devices=devices,
    )

    return {
        "model": model,
        "trainer": trainer,
        "train_input": train_input,
        "train_target": train_target,
    }
