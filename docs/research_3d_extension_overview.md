# Extending VibeSpace from 2D Image Blending to 3D Vibe Blending

## Goal
Preserve the VibeSpace philosophy in 3D:
- reuse strong pretrained backbones,
- train only a lightweight adapter/MLP quickly per task,
- blend only relevant attributes ("vibe") rather than full scene identity.

## What the current 2D pipeline gives us
Current VibeSpace learns a compact latent "vibe" from DINO features and decodes to CLIP features, then uses IP-Adapter-guided diffusion for synthesis.

A practical 3D extension can keep this exact structure:
1. **Pretrained encoder features** -> structured feature tokens for each view/point.
2. **Lightweight vibe mapper (MLP)** -> low-dimensional vibe manifold.
3. **Lightweight decoder** -> conditioning features for a frozen 3D generator.

## Representation options for 3D
Choose one based on compute and data:

1. **Multi-view latent blending (recommended first stage)**
   - Represent a 3D object/scene by a set of calibrated views.
   - Extract per-view features using frozen DINO/CLIP-style encoders.
   - Learn vibe in feature space across all views jointly.
   - Decode to conditioning vectors for a frozen multi-view diffusion or image-to-3D model.

2. **Point/voxel/Gaussian feature blending (second stage)**
   - Use a canonical point cloud / 3D Gaussians / triplanes with features.
   - Learn vibe space on local 3D features directly.
   - Decode to renderer/generator conditions.

3. **NeRF/triplane latent blending (advanced stage)**
   - Use latent planes or radiance-field features from a pretrained 3D backbone.
   - Blend in latent space and render novel views.

## Base model choices (freeze-heavy, train-light)
- **Geometry-aware priors**: DUSt3R, MASt3R, VGGT, Depth Anything, Segment Anything for correspondences/masks.
- **3D generators**: LRM/Instant3D-style feed-forward reconstructor, Zero-1-to-3 family, triplane diffusion backbones, Gaussian Splatting priors.
- **Text-image feature priors**: DINOv2, SigLIP/CLIP for semantic alignment.

Keep these frozen and only train:
- a small vibe encoder/decoder MLP,
- optionally tiny LoRA on a cross-attention block if quality needs help.

## Core research idea: 3D vibe as correspondence-aware feature direction
In 2D VibeSpace, correspondences and clustering find relevant local directions. In 3D, do the same but with 3D-consistent correspondences:

1. Build cross-instance correspondences between two assets (A,B):
   - multi-view feature matching + epipolar/depth consistency,
   - optionally lift to point-level matches in canonical space.
2. Cluster matched regions into semantic parts (e.g., handle/body/legs/wings).
3. Compute per-part direction vectors in latent feature space.
4. Interpolate only on matched part subspaces to avoid identity leakage.

This yields "vibe transfer" (shared attributes) rather than full shape morphing.

## Minimal training loop (prototype)
1. Input: two 3D assets or two image sets with poses.
2. Extract frozen features for each view/point.
3. Build matched part graph and part-level embeddings.
4. Train tiny MLP:
   - encoder: high-dim feature -> low-dim vibe,
   - decoder: vibe -> generator conditioning feature.
5. Losses (small-batch friendly):
   - reconstruction loss on target conditioning features,
   - structure loss (NCut/graph Laplacian consistency, as in current VibeSpace spirit),
   - view-consistency loss across rendered novel views,
   - optional geometry consistency (depth/normal regularization).
6. Inference: interpolate/extrapolate vibe coefficients and render outputs.

## Dataset strategy
Start with object-centric categories for controllability:
- Objaverse-LVIS, ABO, ShapeNet, GSO.

For each training/eval pair:
- obtain 8-24 rendered views with known intrinsics/extrinsics,
- optionally include text tags to evaluate semantic retention.

## Evaluation plan
Quantitative:
- CLIP/SigLIP similarity to source prompts/attributes,
- DINO feature preservation for non-target attributes,
- geometry metrics (Chamfer/F-score/normal consistency) if GT geometry exists,
- multi-view consistency metrics.

Human/creative evaluation:
- pairwise preference on "vibe captured" vs "identity preserved",
- attribute disentanglement scores using positive/negative vibe controls.

## Suggested milestone roadmap
1. **M1 (2-4 weeks)**: multi-view 2.5D prototype
   - input two objects (multi-view images),
   - learn vibe MLP on per-view features,
   - decode to frozen image generator per view,
   - enforce cross-view consistency.
2. **M2 (4-8 weeks)**: explicit 3D latent support
   - shift from per-view to triplane/Gaussian latent blending.
3. **M3 (8-12 weeks)**: negative vibe + extrapolation in 3D
   - suppress unwanted attributes with negative examples,
   - enable controlled extrapolation beyond endpoints.

## Risks and mitigations
- **Unstable correspondences across views** -> use robust multi-view matching + confidence filtering.
- **Geometry drift during interpolation** -> regularize with depth/normal/cycle consistency.
- **Overfitting with tiny pair data** -> keep model tiny, add synthetic augmentations, stop early.
- **Attribute leakage** -> part-masked blending and negative vibe subtraction.

## Recommended first implementation in this repo
1. Keep existing vibe MLP code path conceptually intact.
2. Replace single-image feature extraction with multi-view batch extraction.
3. Add a correspondence module returning part assignments across views/assets.
4. Train on stacked multi-view tokens while preserving a lightweight MLP-only update.
5. Swap 2D IP-Adapter decoder target with a 3D-capable conditioning target (multi-view diffusion or triplane decoder input).

This keeps the original VibeSpace spirit: **pretrained models + lightweight fast training + semantically targeted blending**.
