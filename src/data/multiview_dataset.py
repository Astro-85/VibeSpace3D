"""Multi-view dataset utilities for VibeSpace3D Week-1 infrastructure.

This module defines a lightweight manifest-based dataset format and tensorization
helpers for calibrated multi-view samples.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class ViewRecord:
    image_path: Path
    intrinsics: torch.Tensor  # (3, 3)
    pose_wc: torch.Tensor  # (4, 4)
    depth_path: Optional[Path] = None
    normal_path: Optional[Path] = None
    mask_path: Optional[Path] = None


@dataclass
class MultiViewSample:
    object_id: str
    views: List[ViewRecord]
    text: Optional[str] = None


class MultiViewDataset(Dataset):
    """Load multi-view samples from a JSON manifest.

    Manifest schema:
    {
      "samples": [
        {
          "object_id": "...",
          "text": "optional",
          "views": [
            {
              "image": "relative/or/absolute/path.png",
              "K": [[...], [...], [...]],
              "T_wc": [[...], [...], [...], [...]],
              "depth": "optional/path",
              "normal": "optional/path",
              "mask": "optional/path"
            }
          ]
        }
      ]
    }
    """

    def __init__(self, manifest_path: str | Path, min_views: int = 2):
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.min_views = min_views

        with self.manifest_path.open("r", encoding="utf-8") as f:
            raw_manifest: Dict[str, Any] = json.load(f)

        raw_samples = raw_manifest.get("samples", [])
        self.samples: List[MultiViewSample] = [self._parse_sample(s) for s in raw_samples]

    def _resolve(self, maybe_path: Optional[str]) -> Optional[Path]:
        if maybe_path is None:
            return None
        p = Path(maybe_path)
        return p if p.is_absolute() else self.root / p

    def _parse_sample(self, raw: Dict[str, Any]) -> MultiViewSample:
        views: List[ViewRecord] = []
        for view in raw.get("views", []):
            k = torch.tensor(view["K"], dtype=torch.float32)
            t_wc = torch.tensor(view["T_wc"], dtype=torch.float32)
            views.append(
                ViewRecord(
                    image_path=self._resolve(view["image"]),
                    intrinsics=k,
                    pose_wc=t_wc,
                    depth_path=self._resolve(view.get("depth")),
                    normal_path=self._resolve(view.get("normal")),
                    mask_path=self._resolve(view.get("mask")),
                )
            )

        if len(views) < self.min_views:
            raise ValueError(
                f"Sample {raw.get('object_id', '<unknown>')} has {len(views)} views, "
                f"expected >= {self.min_views}."
            )

        return MultiViewSample(
            object_id=raw["object_id"],
            text=raw.get("text"),
            views=views,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> MultiViewSample:
        return self.samples[idx]


def tensorize_multiview_sample(
    sample: MultiViewSample,
    image_transform,
    max_views: Optional[int] = None,
) -> Dict[str, torch.Tensor | str]:
    """Convert a multi-view sample into stacked tensors."""
    selected_views: Sequence[ViewRecord] = sample.views if max_views is None else sample.views[:max_views]

    images = []
    intrinsics = []
    poses = []
    for v in selected_views:
        with Image.open(v.image_path) as img:
            images.append(image_transform(img.convert("RGB")))
        intrinsics.append(v.intrinsics)
        poses.append(v.pose_wc)

    image_tensor = torch.stack(images, dim=0)
    intrinsics_tensor = torch.stack(intrinsics, dim=0)
    poses_tensor = torch.stack(poses, dim=0)

    output: Dict[str, torch.Tensor | str] = {
        "object_id": sample.object_id,
        "images": image_tensor,
        "intrinsics": intrinsics_tensor,
        "poses_wc": poses_tensor,
    }
    if sample.text is not None:
        output["text"] = sample.text
    return output
