"""Multi-view feature extraction wrappers for VibeSpace3D.

Week-1 goal: provide a thin bridge from MultiViewSample tensors to frozen
feature backbones already used in the repository.
"""

from __future__ import annotations

from typing import Dict

import torch

from src.extract_features import (
    clip_image_transform,
    dino_image_transform,
    extract_clip_features,
    extract_dino_features,
)


def extract_multiview_backbone_features(
    images: torch.Tensor,
    ipadapter_version: str = "sd15",
) -> Dict[str, torch.Tensor]:
    """Extract DINO + CLIP features from stacked multi-view images.

    Args:
        images: Tensor of shape (V, C, H, W)
        ipadapter_version: SD backbone version for CLIP embedding target

    Returns:
        Dict with:
            dino_features: (V, L_dino, D_dino)
            clip_features: (V, L_clip, D_clip)
    """
    if images.dim() != 4:
        raise ValueError(f"Expected (V, C, H, W), got {tuple(images.shape)}")

    dino_features = extract_dino_features(images)

    # CLIP transform expects PIL in normal flow. For Week-1 infra we assume
    # callers provide already transformed tensors at CLIP resolution if needed.
    # If resolution mismatches, torch interpolation can be added in Week-2.
    clip_features = extract_clip_features(images, ipadapter_version=ipadapter_version)

    return {
        "dino_features": dino_features,
        "clip_features": clip_features,
    }


def apply_multiview_transforms(pil_images):
    """Apply repo-standard transforms to a list of PIL views."""
    dino_images = torch.stack([dino_image_transform(img) for img in pil_images], dim=0)
    clip_images = torch.stack([clip_image_transform(img) for img in pil_images], dim=0)
    return {"dino_images": dino_images, "clip_images": clip_images}
