# VibeSpace3D First Implementation Plan

This document turns the 3D overview into an **actionable implementation plan** for two parallel prototype tracks:

- **Track A**: multi-view diffusion conditioning (fastest path to visible results)
- **Track B**: triplane latent decoding (stronger explicit 3D representation)

Both tracks keep the core VibeSpace principle:

1. frozen pretrained feature extractors/generators,
2. fast per-task training of lightweight MLP adapters,
3. blending only correspondence-relevant attributes ("vibe").

---

## 0) Current baseline to preserve

From the current repo design:

- feature extraction: DINO features + CLIP targets,
- lightweight encoder/decoder MLP (`VibeSpaceModel`),
- structure-aware losses (NCut-inspired + reconstruction),
- interpolation by correspondence-guided local direction field.

The first implementation should **reuse this structure**, changing only:

- input modality: single image -> multi-view set,
- decoder target: 2D CLIP conditioning -> 3D-capable conditioning.

---

## 1) Shared data/interface layer (both tracks)

### 1.1 Data format (minimum)

Create a unified sample format:

```text
sample = {
  object_id: str,
  views: [
    {
      image: HxWx3,
      K: 3x3 intrinsics,
      T_wc: 4x4 pose,
      optional depth/normal/mask
    }, ...
  ],
  text: optional caption/tags
}
```

### 1.2 New module boundaries

Add these modules first (stubs acceptable initially):

- `src/data/multiview_dataset.py`
  - load paired objects and `N` calibrated views
- `src/features/multiview_features.py`
  - batch extraction of frozen DINO/CLIP/SigLIP features
- `src/correspondence/multiview_correspondence.py`
  - correspondence graph + confidence scores
- `src/training/vibespace3d_trainer.py`
  - shared loop for both Track A/B

### 1.3 Shared vibe adapter

Implement `VibeSpace3DModel` as a minimal extension of current `VibeSpaceModel`:

- `encoder_mlp`: high-dim local token -> `vibe_dim`
- `decoder_mlp`: `vibe_dim` -> target conditioning dim
- optional learned view token / pose token concatenation

Keep parameter count small (roughly comparable to current MLP scale).

---

## 2) Track A: Multi-view diffusion conditioning (recommended to start)

## 2.1 Objective

Given source pair `(A, B)` with multi-view observations:

- learn a local direction field in vibe space for matched regions,
- decode blended vibe tokens into conditioning tokens,
- run frozen multi-view-consistent diffusion/image generator per view.

### 2.2 Conditioning target choices

Pick one practical target:

1. CLIP-like image condition tokens per view (simplest), or
2. cross-attention key/value adapter tokens if model API allows.

Use frozen generator; avoid full finetuning.

### 2.3 Losses (initial)

Use a lightweight loss stack:

- `L_recon`: decode to target conditioning features (L1/Huber)
- `L_struct`: preserve neighborhood structure in compressed vibe space
- `L_view_cons`: rendered/intermediate output consistency across adjacent views
- optional `L_neg`: subtract negative vibe directions when negative pair supplied

Total:

```text
L = w_recon*L_recon + w_struct*L_struct + w_view*L_view_cons + w_neg*L_neg
```

### 2.4 Milestone A deliverables

- train on 2 objects x 8-12 views each,
- generate interpolation sweep at fixed camera ring,
- show view-consistent transitions and part-selective blending.

---

## 3) Track B: Triplane latent decoding (explicit 3D track)

### 3.1 Objective

Replace per-view conditioning target with a **canonical triplane latent target**:

- encode observations into triplane features (frozen 3D backbone),
- perform vibe blending in local feature space,
- decode to triplane latent and render novel views.

### 3.2 Canonicalization + correspondences

Key requirement: stable canonical frame.

Use one of:

- pose-normalized object frame (if available), or
- learned canonicalization from frozen reconstructor outputs.

Then compute correspondences at one of three levels:

1. image-space matched pixels -> lifted to 3D points,
2. point correspondences -> triplane cell indices,
3. directly match triplane feature cells by nearest-neighbor + confidence.

### 3.3 Losses (initial)

- `L_triplane_recon`: decode vibe -> triplane latent target
- `L_render`: photometric/perceptual loss on rendered supervision views
- `L_geom`: depth/normal consistency regularization
- `L_cycle`: optional blend cycle consistency (A->B->A)

Total:

```text
L = w_tri*L_triplane_recon + w_render*L_render + w_geom*L_geom + w_cycle*L_cycle
```

### 3.4 Milestone B deliverables

- novel-view render stability under interpolation,
- geometry drift reduced vs Track A,
- qualitative part transfer with less identity leakage.

---

## 4) 6-week execution sequence

### Week 1: infrastructure

- dataset + multiview dataloader + feature cache
- shared adapter class + config schema

### Week 2: correspondence MVP

- nearest-neighbor feature matching + confidence pruning
- part clustering with simple KMeans/NCut over correspondences

### Week 3: Track A first train/infer

- train on tiny benchmark pairs
- generate interpolation grid and debug failure modes

### Week 4: Track A stabilization

- add view-consistency loss
- add negative-vibe control path

### Week 5: Track B scaffold

- triplane target extraction pipeline
- adapter decode to triplane latent

### Week 6: Track B baseline

- render-time evaluation
- compare Track A vs Track B on consistency and controllability

---

## 5) Evaluation protocol for first implementation

### Quantitative

- semantic retention: CLIP/SigLIP similarity to intended attributes
- content preservation: DINO feature similarity for non-target regions
- view consistency: LPIPS/feature variance across adjacent viewpoints
- geometry consistency (Track B): depth/normal disagreement metrics

### Human study (small)

3 questions per sample:

1. "Is the intended vibe transferred?"
2. "Is source identity over-copied?" (lower is better)
3. "Are views geometrically coherent?"

---

## 6) Risk register (first implementation)

- **Noisy correspondences** -> confidence threshold + mutual NN + cycle filtering
- **View flicker in Track A** -> stronger view-consistency constraints and shared random seed controls
- **Canonical misalignment in Track B** -> add pose-normalization checks and fail-fast diagnostics
- **Overfitting to pair** -> early stopping + simple augmentations + tiny adapter only

---

## 7) Immediate next PR breakdown

PR-1 (infra):
- add multiview dataset/feature modules and config entries
- no generator integration yet

PR-2 (Track A MVP):
- correspondence + direction-field in multiview token space
- decode to per-view conditioning + inference script

PR-3 (Track A quality):
- view consistency + negative vibe controls

PR-4 (Track B MVP):
- triplane target extraction + decode + renderer bridge

This sequence keeps changes reviewable while producing visible outputs early.
