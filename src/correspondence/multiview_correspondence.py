"""Week-1 multi-view correspondence scaffolding.

Implements a simple mutual-nearest-neighbor matcher between token features
from two views/objects with confidence filtering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass
class CorrespondenceResult:
    src_indices: torch.Tensor
    dst_indices: torch.Tensor
    confidences: torch.Tensor


@torch.no_grad()
def mutual_nn_correspondence(
    src_features: torch.Tensor,
    dst_features: torch.Tensor,
    min_confidence: float = 0.2,
) -> CorrespondenceResult:
    """Compute mutual nearest-neighbor matches between token features.

    Args:
        src_features: (N, D)
        dst_features: (M, D)
        min_confidence: cosine similarity threshold
    """
    if src_features.dim() != 2 or dst_features.dim() != 2:
        raise ValueError("Expected 2D feature tensors: (N, D) and (M, D)")

    src = F.normalize(src_features, dim=-1)
    dst = F.normalize(dst_features, dim=-1)

    sim = src @ dst.T  # (N, M)

    src_to_dst = sim.argmax(dim=1)
    dst_to_src = sim.argmax(dim=0)

    src_ids = torch.arange(src.shape[0], device=src.device)
    mutual = dst_to_src[src_to_dst] == src_ids

    conf = sim[src_ids, src_to_dst]
    keep = mutual & (conf >= min_confidence)

    return CorrespondenceResult(
        src_indices=src_ids[keep].cpu(),
        dst_indices=src_to_dst[keep].cpu(),
        confidences=conf[keep].cpu(),
    )


@torch.no_grad()
def build_pairwise_view_correspondence(
    a_view_features: torch.Tensor,
    b_view_features: torch.Tensor,
    min_confidence: float = 0.2,
) -> Dict[str, CorrespondenceResult]:
    """Build correspondences for every view pair between two objects.

    Args:
        a_view_features: (V_a, T, D)
        b_view_features: (V_b, T, D)
    """
    if a_view_features.dim() != 3 or b_view_features.dim() != 3:
        raise ValueError("Expected tensors shaped (V, T, D)")

    results: Dict[str, CorrespondenceResult] = {}
    for i in range(a_view_features.shape[0]):
        for j in range(b_view_features.shape[0]):
            key = f"a{i}_b{j}"
            results[key] = mutual_nn_correspondence(
                a_view_features[i],
                b_view_features[j],
                min_confidence=min_confidence,
            )
    return results
