# powerline-geometry-perception

English | [简体中文](./README_zh-CN.md)

A geometry-oriented visual perception frontend for power line localization and local line-direction understanding, built on top of **mmsegmentation**.

## Overview

This project targets **power-line geometric perception** rather than generic object detection. Instead of predicting bounding boxes, the current mainline predicts:

- **where the power line centerline is**
- **which direction the line is locally heading**

The project started from a lightweight **ResNet18 + FPN** two-head baseline and has evolved into a stronger **V2** pipeline based on **SegFormer-B1**, with **distance-field auxiliary supervision**, **sliding-window inference** for large TTPLA images, and an optional **test-time geometric verifier**.

## Why this project exists

Power lines are:

- extremely thin
- long-range and topology-sensitive
- easily confused with other bright linear structures such as road markings, building edges, poles, fences, and tower boundaries

For this reason, the project is intentionally designed as a **geometry perception frontend** instead of a standard detector.

## Evolution

### V1

- Backbone: **ResNet18**
- Neck: **FPN**
- Fusion: **SharedFusion**
- Heads:
  - CenterHead
  - OrientationHead
- Training style: patch training on large TTPLA images

### V2

- Backbone upgraded to **SegFormer-B1**
- `neck=None` in the mainline (no fixed FPN dependency)
- SharedFusion adapted for transformer multi-scale features
- Added **DistanceFieldHead** as auxiliary geometric supervision
- Added **slide inference** for large-image testing
- Added optional **Geometric Verifier** for test-time false-line suppression

## Key design principles

1. **Two-head geometry modeling**
   - centerline position and line direction are predicted in parallel
2. **Patch training for thin structures**
   - large TTPLA images are cropped into high-resolution patches
3. **Geometry-aware supervision**
   - orientation is supervised only on valid centerline regions
   - distance map is generated online from the center mask
4. **Geometry-aware test-time filtering**
   - an optional verifier can suppress short, isolated, and off-direction false segments

## Dataset and labels

The project is currently built around **TTPLA**.

The internal training targets are organized as:

- `center/*.png` — binary centerline maps
- `orient/*.npy` — local 2D orientation fields
- `images/*` — original RGB images

For V2 with distance auxiliary supervision:

- `gt_distance_map` is **generated online** from the centerline mask during loading
- no extra manual distance annotation is required

## Repository structure

```text
configs/
  powerline_v1/
projects/
  powerline_v1/
    datasets/
    models/
      heads/
      modules/
      segmentors/
    utils/
tools/
  train.py
  test.py
  debug/
work_dirs/
```

## Example commands

### V2 + distance auxiliary training

```bash
python tools/train.py configs/powerline_v1/powerline_v1_segformer_b1_aux_distance.py
```

### Debug training

```bash
python tools/train.py configs/powerline_v1/powerline_v1_segformer_b1_aux_distance_debug.py
```

### Slide test on large images

```bash
python tools/test.py \
  configs/powerline_v1/powerline_v1_segformer_b1_aux_distance_test_slide.py \
  work_dirs/powerline_v1_segformer_b1_aux_distance/iter_20000.pth
```

### Slide test with saved visualizations

```bash
python tools/test.py \
  configs/powerline_v1/powerline_v1_segformer_b1_aux_distance_test_slide.py \
  work_dirs/powerline_v1_segformer_b1_aux_distance/iter_20000.pth \
  --show-dir work_dirs/powerline_v1_segformer_b1_aux_distance/test_vis_iter_20000
```

### Auxiliary target / prediction inspection

```bash
python tools/debug/inspect_aux_targets_and_preds.py \
  --config configs/powerline_v1/powerline_v1_segformer_b1_aux_distance_debug.py \
  --checkpoint work_dirs/.../iter_20000.pth \
  --num-samples 8 \
  --out-dir work_dirs/debug_vis
```

## What the geometric verifier does

The optional verifier is a **test-time hard geometric filter**.

It uses:

- centerline probability map
- local orientation field
- optional distance prediction

and performs component-level filtering based on:

- area / estimated length / aspect ratio
- center confidence
- orientation consistency
- dominant direction clustering
- optional isolation filtering

Its purpose is to reduce short, isolated, and off-direction false line segments that survive the neural prediction stage.

## Current status

### Stable mainline

- ResNet18 + FPN V1 baseline: completed
- SegFormer-B1 V2 mainline: completed
- Distance auxiliary supervision: completed and trainable
- Slide inference for large test images: completed
- Geometric verifier: completed as a test-time post-filter

### In-progress / exploratory

- stronger branch-level / skeleton-level geometric verifier
- harder false-positive suppression on complex backgrounds
- hard-negative automatic mining / reinjection
- attraction-field or richer geometric auxiliary supervision
- integration with downstream autonomous flight / inspection stack

### Not yet enabled in the current mainline

- **Tower auxiliary head**: code skeleton exists, but the tower data path is not yet fully connected

## What is NOT the goal

This project is **not** trying to be a generic YOLO-style detector or a standard coarse semantic segmentation model.

The goal is a **power-line geometry perception frontend** that can later support:

- line-following UAV flight
- inspection-aware perception
- geometry-constrained downstream planning and control

## Roadmap

- improve geometric verifier from component-level filtering to branch-level / skeleton-level pruning
- add hard-negative mining for confusing linear distractors
- investigate richer geometric supervision beyond the current distance field
- close the sim-to-real loop for UAV deployment
- integrate with a PPO-based policy for real flight and inspection tasks

## Acknowledgements

- [OpenMMLab / mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
- TTPLA-related dataset and power-line perception research community

## Notes

This repository is under active development. Some modules are stable mainline components, while others are research branches or engineering placeholders.
