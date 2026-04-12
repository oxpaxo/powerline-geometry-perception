#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Faster TTPLA converter:
polygon -> line_mask
polygon -> centerline polyline -> center.png
polygon -> tangent -> orient.npy

Main fixes vs previous version:
1. orientation rasterization only updates local ROI per segment, not full image
2. per-sample progress logging
3. output directories/files are written sample by sample
4. optional --no-orient to debug speed quickly
"""

import argparse
import base64
import json
import math
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw


VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_stem(name: str) -> str:
    return Path(name).stem.strip()


def infer_image_path(json_path: Path, image_path_value: Optional[str]) -> Optional[Path]:
    if image_path_value:
        candidate = json_path.parent / image_path_value
        if candidate.exists():
            return candidate

    stem = json_path.stem
    for ext in VALID_IMAGE_EXTS:
        candidate = json_path.with_suffix(ext)
        if candidate.exists():
            return candidate

    for p in json_path.parent.iterdir():
        if p.is_file() and p.stem == stem and p.suffix.lower() in VALID_IMAGE_EXTS:
            return p

    return None


def read_labelme_json(json_path: Path) -> Dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def decode_image_size_from_imageData(image_data_b64: str) -> Tuple[int, int]:
    raw = base64.b64decode(image_data_b64)
    img_array = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode imageData from LabelMe JSON.")
    h, w = img.shape[:2]
    return h, w


def get_image_hw(meta: Dict, image_path: Optional[Path]) -> Tuple[int, int]:
    h = meta.get("imageHeight", None)
    w = meta.get("imageWidth", None)
    if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
        return h, w

    image_data = meta.get("imageData", None)
    if image_data:
        return decode_image_size_from_imageData(image_data)

    if image_path is not None:
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")
        h, w = img.shape[:2]
        return h, w

    raise ValueError("Cannot determine image size from JSON or paired image file.")


def polygon_to_mask(
    image_size_hw: Tuple[int, int],
    polygons: Sequence[Sequence[Tuple[float, float]]]
) -> np.ndarray:
    h, w = image_size_hw
    canvas = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(canvas)
    for poly in polygons:
        if poly is None or len(poly) < 3:
            continue
        xy = [(float(x), float(y)) for x, y in poly]
        draw.polygon(xy=xy, outline=1, fill=1)
    return np.array(canvas, dtype=np.uint8) * 255


def collect_cable_shapes(meta: Dict, cable_labels: Iterable[str]) -> List[Dict[str, object]]:
    cable_labels = {x.lower() for x in cable_labels}
    items: List[Dict[str, object]] = []

    for shape in meta.get("shapes", []):
        label = str(shape.get("label", "")).strip().lower()
        shape_type = str(shape.get("shape_type", "")).strip().lower()
        points = shape.get("points", [])

        if label not in cable_labels:
            continue
        if shape_type not in {"polygon", "linestrip"}:
            continue
        if not isinstance(points, list) or len(points) < 2:
            continue

        pts = []
        for pt in points:
            if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                continue
            pts.append((float(pt[0]), float(pt[1])))

        if shape_type == "polygon" and len(pts) >= 3:
            items.append({"shape_type": "polygon", "points": pts})
        elif shape_type == "linestrip" and len(pts) >= 2:
            items.append({"shape_type": "linestrip", "points": pts})

    return items


def save_png(mask_uint8: np.ndarray, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    ok = cv2.imwrite(str(out_path), mask_uint8)
    if not ok:
        raise IOError(f"Failed to write PNG: {out_path}")


def copy_image(src_image_path: Path, dst_image_path: Path) -> None:
    ensure_dir(dst_image_path.parent)
    shutil.copy2(src_image_path, dst_image_path)


def maybe_resize_if_needed(img: np.ndarray, size: Optional[Tuple[int, int]]) -> np.ndarray:
    if size is None:
        return img
    target_w, target_h = size
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def maybe_resize_mask_if_needed(mask: np.ndarray, size: Optional[Tuple[int, int]]) -> np.ndarray:
    if size is None:
        return mask
    target_w, target_h = size
    return cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)


def resize_center_mask_antialiased(mask: np.ndarray, size: Optional[Tuple[int, int]], threshold: int) -> np.ndarray:
    if size is None:
        return mask
    target_w, target_h = size
    resized = cv2.resize(mask.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return np.where(resized >= float(threshold), 255, 0).astype(np.uint8)


def resize_orient_field(orient: np.ndarray, center_mask_small: np.ndarray) -> np.ndarray:
    h, w = center_mask_small.shape
    ox = cv2.resize(orient[..., 0], (w, h), interpolation=cv2.INTER_LINEAR)
    oy = cv2.resize(orient[..., 1], (w, h), interpolation=cv2.INTER_LINEAR)
    norm = np.sqrt(ox * ox + oy * oy) + 1e-6
    ox = ox / norm
    oy = oy / norm
    out = np.zeros((h, w, 2), dtype=np.float32)
    valid = center_mask_small > 0
    out[..., 0][valid] = ox[valid]
    out[..., 1][valid] = oy[valid]
    return out


def close_polygon_ring(points: Sequence[Tuple[float, float]]) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] < 3:
        raise ValueError("Polygon needs at least 3 points.")
    if np.linalg.norm(pts[0] - pts[-1]) > 1e-6:
        pts = np.vstack([pts, pts[0]])
    return pts


def polyline_length(pts: np.ndarray) -> float:
    if pts.shape[0] < 2:
        return 0.0
    seg = np.diff(pts, axis=0)
    return float(np.linalg.norm(seg, axis=1).sum())


def resample_polyline(pts: np.ndarray, num_points: int) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if pts.shape[0] == 1:
        return np.repeat(pts, num_points, axis=0)
    if num_points <= 2:
        return np.vstack([pts[0], pts[-1]]).astype(np.float32)

    seg = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cum[-1])

    if total <= 1e-6:
        return np.repeat(pts[:1], num_points, axis=0)

    targets = np.linspace(0.0, total, num_points, dtype=np.float32)
    out = np.zeros((num_points, 2), dtype=np.float32)

    j = 0
    for i, t in enumerate(targets):
        while j + 1 < len(cum) and cum[j + 1] < t:
            j += 1
        if j >= len(seg_len):
            out[i] = pts[-1]
            continue
        den = max(float(seg_len[j]), 1e-6)
        alpha = (float(t) - float(cum[j])) / den
        alpha = min(max(alpha, 0.0), 1.0)
        out[i] = (1.0 - alpha) * pts[j] + alpha * pts[j + 1]
    return out


def split_closed_ring_by_indices(ring: np.ndarray, idx_a: int, idx_b: int) -> Tuple[np.ndarray, np.ndarray]:
    n = ring.shape[0] - 1
    idx_a = int(idx_a) % n
    idx_b = int(idx_b) % n
    if idx_a == idx_b:
        raise ValueError("Split indices must be different.")
    if idx_a > idx_b:
        idx_a, idx_b = idx_b, idx_a

    base = ring[:-1]
    path1 = base[idx_a:idx_b + 1]
    path2 = np.vstack([base[idx_b:], base[:idx_a + 1]])
    return path1.astype(np.float32), path2.astype(np.float32)


def moving_average_polyline(pts: np.ndarray, win: int) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape[0] < 3 or win <= 1:
        return pts
    if win % 2 == 0:
        win += 1
    pad = win // 2
    padded = np.pad(pts, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(win, dtype=np.float32) / float(win)
    out_x = np.convolve(padded[:, 0], kernel, mode="valid")
    out_y = np.convolve(padded[:, 1], kernel, mode="valid")
    out = np.stack([out_x, out_y], axis=1).astype(np.float32)
    out[0] = pts[0]
    out[-1] = pts[-1]
    return out


def remove_duplicate_consecutive_points(pts: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape[0] <= 1:
        return pts
    keep = [0]
    for i in range(1, pts.shape[0]):
        if float(np.linalg.norm(pts[i] - pts[keep[-1]])) > eps:
            keep.append(i)
    return pts[keep]


def compute_polygon_centerline(
    polygon: Sequence[Tuple[float, float]],
    samples_per_px: float = 0.35,
    min_samples: int = 24,
    smooth_window: int = 5,
) -> Optional[np.ndarray]:
    ring = close_polygon_ring(polygon)
    base = ring[:-1]
    n = base.shape[0]
    if n < 4:
        return None

    mean = base.mean(axis=0, keepdims=True)
    centered = base - mean
    cov = centered.T @ centered / max(n - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, int(np.argmax(eigvals))]
    proj = centered @ major

    idx_min = int(np.argmin(proj))
    idx_max = int(np.argmax(proj))
    if idx_min == idx_max:
        return None

    edge1, edge2 = split_closed_ring_by_indices(ring, idx_min, idx_max)
    len1 = polyline_length(edge1)
    len2 = polyline_length(edge2)
    if len1 <= 1e-4 or len2 <= 1e-4:
        return None

    num_samples = int(max(min_samples, round(max(len1, len2) * float(samples_per_px))))
    num_samples = max(num_samples, 8)

    s1 = resample_polyline(edge1, num_samples)
    s2 = resample_polyline(edge2, num_samples)

    if float(np.mean(np.linalg.norm(s1 - s2[::-1], axis=1))) < float(np.mean(np.linalg.norm(s1 - s2, axis=1))):
        s2 = s2[::-1].copy()

    center = 0.5 * (s1 + s2)
    center = moving_average_polyline(remove_duplicate_consecutive_points(center), smooth_window)
    center = remove_duplicate_consecutive_points(center)

    if center.shape[0] < 2 or polyline_length(center) <= 1e-3:
        return None
    return center.astype(np.float32)


def linestrip_to_centerline(
    points: Sequence[Tuple[float, float]],
    samples_per_px: float = 0.35,
    min_samples: int = 24,
    smooth_window: int = 3,
) -> Optional[np.ndarray]:
    pts = remove_duplicate_consecutive_points(np.asarray(points, dtype=np.float32))
    if pts.shape[0] < 2:
        return None
    length = polyline_length(pts)
    num_samples = int(max(min_samples, round(length * float(samples_per_px))))
    num_samples = max(num_samples, 8)
    center = resample_polyline(pts, num_samples)
    center = moving_average_polyline(center, smooth_window)
    center = remove_duplicate_consecutive_points(center)
    if center.shape[0] < 2:
        return None
    return center.astype(np.float32)


def draw_polyline_mask(
    canvas_hw: Tuple[int, int],
    polylines: Sequence[np.ndarray],
    thickness: int,
    antialias: bool = True,
) -> np.ndarray:
    h, w = canvas_hw
    canvas = np.zeros((h, w), dtype=np.uint8)
    line_type = cv2.LINE_AA if antialias else cv2.LINE_8
    for line in polylines:
        if line is None or len(line) < 2:
            continue
        pts = np.round(np.asarray(line, dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], isClosed=False, color=255, thickness=int(max(1, thickness)), lineType=line_type)
    return canvas


def build_orient_from_centerlines_fast(
    canvas_hw: Tuple[int, int],
    centerlines: Sequence[np.ndarray],
    thickness: int,
) -> np.ndarray:
    h, w = canvas_hw
    acc_x = np.zeros((h, w), dtype=np.float32)
    acc_y = np.zeros((h, w), dtype=np.float32)
    acc_w = np.zeros((h, w), dtype=np.float32)
    pad = int(max(2, thickness + 2))

    for line in centerlines:
        if line is None or len(line) < 2:
            continue
        pts = np.asarray(line, dtype=np.float32)

        for i in range(len(pts) - 1):
            p0 = pts[i]
            p1 = pts[i + 1]
            dx = float(p1[0] - p0[0])
            dy = float(p1[1] - p0[1])
            norm = math.hypot(dx, dy)
            if norm <= 1e-6:
                continue

            vx = dx / norm
            vy = dy / norm
            if vy < 0:
                vx, vy = -vx, -vy

            minx = max(0, int(math.floor(min(p0[0], p1[0]) - pad)))
            maxx = min(w - 1, int(math.ceil(max(p0[0], p1[0]) + pad)))
            miny = max(0, int(math.floor(min(p0[1], p1[1]) - pad)))
            maxy = min(h - 1, int(math.ceil(max(p0[1], p1[1]) + pad)))
            if minx > maxx or miny > maxy:
                continue

            roi_h = maxy - miny + 1
            roi_w = maxx - minx + 1
            tmp = np.zeros((roi_h, roi_w), dtype=np.uint8)

            a = (int(round(p0[0])) - minx, int(round(p0[1])) - miny)
            b = (int(round(p1[0])) - minx, int(round(p1[1])) - miny)
            cv2.line(tmp, a, b, color=255, thickness=int(max(1, thickness)), lineType=cv2.LINE_AA)

            weight = tmp.astype(np.float32) / 255.0
            if not np.any(weight > 0):
                continue

            acc_x[miny:maxy + 1, minx:maxx + 1] += weight * vx
            acc_y[miny:maxy + 1, minx:maxx + 1] += weight * vy
            acc_w[miny:maxy + 1, minx:maxx + 1] += weight

    orient = np.zeros((h, w, 2), dtype=np.float32)
    valid = acc_w > 1e-6
    if np.any(valid):
        ox = np.zeros((h, w), dtype=np.float32)
        oy = np.zeros((h, w), dtype=np.float32)
        ox[valid] = acc_x[valid] / acc_w[valid]
        oy[valid] = acc_y[valid] / acc_w[valid]
        norm = np.sqrt(ox * ox + oy * oy) + 1e-6
        ox[valid] = ox[valid] / norm[valid]
        oy[valid] = oy[valid] / norm[valid]
        orient[..., 0] = ox
        orient[..., 1] = oy
    return orient


def build_center_and_orient_from_shapes(
    image_hw: Tuple[int, int],
    shapes: Sequence[Dict[str, object]],
    center_thickness: int,
    samples_per_px: float,
    min_center_samples: int,
    center_smooth_window: int,
    build_orient: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    centerlines: List[np.ndarray] = []
    polys_for_line_mask: List[List[Tuple[float, float]]] = []

    for item in shapes:
        shape_type = str(item["shape_type"])
        points = item["points"]
        assert isinstance(points, list)

        if shape_type == "polygon":
            polys_for_line_mask.append(points)
            center = compute_polygon_centerline(
                polygon=points,
                samples_per_px=samples_per_px,
                min_samples=min_center_samples,
                smooth_window=center_smooth_window,
            )
            if center is not None and len(center) >= 2:
                centerlines.append(center)
        elif shape_type == "linestrip":
            center = linestrip_to_centerline(
                points=points,
                samples_per_px=samples_per_px,
                min_samples=min_center_samples,
                smooth_window=max(3, center_smooth_window // 2 * 2 + 1),
            )
            if center is not None and len(center) >= 2:
                centerlines.append(center)

    line_mask = polygon_to_mask(image_hw, polys_for_line_mask)
    center_png = draw_polyline_mask(image_hw, centerlines, thickness=center_thickness, antialias=True)

    if build_orient:
        orient = build_orient_from_centerlines_fast(image_hw, centerlines, thickness=center_thickness)
        center_bool = center_png > 0
        orient[..., 0][~center_bool] = 0.0
        orient[..., 1][~center_bool] = 0.0
    else:
        h, w = image_hw
        orient = np.zeros((h, w, 2), dtype=np.float32)

    return line_mask, center_png, orient


def read_split_file(txt_path: Path) -> List[str]:
    stems = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                stems.append(normalize_stem(s))
    return stems


def discover_splits(split_dir: Optional[Path], all_stems: List[str], seed: int = 42) -> Dict[str, List[str]]:
    if split_dir is not None and split_dir.exists():
        train_txt = split_dir / "train.txt"
        validate_txt = split_dir / "validate.txt"
        val_txt = split_dir / "val.txt"
        test_txt = split_dir / "test.txt"

        splits: Dict[str, List[str]] = {}
        if train_txt.exists():
            splits["train"] = read_split_file(train_txt)
        if validate_txt.exists():
            splits["val"] = read_split_file(validate_txt)
        elif val_txt.exists():
            splits["val"] = read_split_file(val_txt)
        if test_txt.exists():
            splits["test"] = read_split_file(test_txt)
        if "train" in splits and "val" in splits:
            return splits

    rng = np.random.default_rng(seed)
    stems = np.array(sorted(all_stems))
    rng.shuffle(stems)
    n = len(stems)
    n_val = max(1, int(round(n * 0.1)))
    val = stems[:n_val].tolist()
    train = stems[n_val:].tolist()
    return {"train": train, "val": val}


def build_index(src_dir: Path) -> Dict[str, Dict]:
    index: Dict[str, Dict] = {}
    for json_path in sorted(src_dir.glob("*.json")):
        meta = read_labelme_json(json_path)
        image_path = infer_image_path(json_path, meta.get("imagePath", None))
        stem = normalize_stem(json_path.name)
        index[stem] = {"stem": stem, "json_path": json_path, "image_path": image_path, "meta": meta}
    return index


def save_split_txt(out_path: Path, stems: Sequence[str], image_ext_map: Dict[str, str]) -> None:
    ensure_dir(out_path.parent)
    with open(out_path, "w", encoding="utf-8") as f:
        for stem in stems:
            f.write(f"{stem}{image_ext_map.get(stem, '.jpg')}\n")


def convert_one_sample(
    meta: Dict,
    image_path: Path,
    dst_image_path: Path,
    dst_line_mask_path: Path,
    dst_center_path: Path,
    dst_orient_path: Path,
    cable_labels: Sequence[str],
    resize_to: Optional[Tuple[int, int]],
    center_thickness: int,
    center_resize_threshold: int,
    samples_per_px: float,
    min_center_samples: int,
    center_smooth_window: int,
    write_orient: bool,
) -> None:
    h, w = get_image_hw(meta, image_path)
    shapes = collect_cable_shapes(meta, cable_labels)

    line_mask, center_png, orient = build_center_and_orient_from_shapes(
        image_hw=(h, w),
        shapes=shapes,
        center_thickness=center_thickness,
        samples_per_px=samples_per_px,
        min_center_samples=min_center_samples,
        center_smooth_window=center_smooth_window,
        build_orient=write_orient,
    )

    if resize_to is not None:
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        img_resized = maybe_resize_if_needed(img, resize_to)
        ensure_dir(dst_image_path.parent)
        if not cv2.imwrite(str(dst_image_path), img_resized):
            raise IOError(f"Failed to write image: {dst_image_path}")

        line_mask = maybe_resize_mask_if_needed(line_mask, resize_to)
        center_small = resize_center_mask_antialiased(center_png, resize_to, threshold=center_resize_threshold)
        center_png = center_small
        if write_orient:
            orient = resize_orient_field(orient, center_small)
        else:
            orient = np.zeros((resize_to[1], resize_to[0], 2), dtype=np.float32)
    else:
        copy_image(image_path, dst_image_path)

    save_png(line_mask, dst_line_mask_path)
    save_png(center_png, dst_center_path)

    if write_orient:
        ensure_dir(dst_orient_path.parent)
        np.save(str(dst_orient_path), orient.astype(np.float32))


def main():
    parser = argparse.ArgumentParser(description="Fast TTPLA polygon-centerline converter")
    parser.add_argument("--src-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--split-dir", type=str, default="")
    parser.add_argument("--labels", nargs="+", default=["cable"])
    parser.add_argument("--resize-width", type=int, default=0)
    parser.add_argument("--resize-height", type=int, default=0)
    parser.add_argument("--center-thickness", type=int, default=7)
    parser.add_argument("--center-resize-threshold", type=int, default=24)
    parser.add_argument("--samples-per-px", type=float, default=0.35)
    parser.add_argument("--min-center-samples", type=int, default=24)
    parser.add_argument("--center-smooth-window", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--no-orient", action="store_true", help="Skip orient generation for debugging speed.")
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    split_dir = Path(args.split_dir) if args.split_dir else None
    ensure_dir(out_dir)

    resize_to = None
    if args.resize_width > 0 and args.resize_height > 0:
        resize_to = (args.resize_width, args.resize_height)

    index = build_index(src_dir)
    if not index:
        raise RuntimeError(f"No JSON files found in {src_dir}")

    missing_images = [stem for stem, rec in index.items() if rec["image_path"] is None]
    for stem in missing_images:
        index.pop(stem, None)

    all_stems = sorted(index.keys())
    if not all_stems:
        raise RuntimeError("No valid image/json pairs remain after filtering.")

    splits = discover_splits(split_dir, all_stems, seed=args.seed)
    for split_name in list(splits.keys()):
        splits[split_name] = [s for s in splits[split_name] if s in index]

    image_ext_map: Dict[str, str] = {}
    for stem, rec in index.items():
        image_ext_map[stem] = rec["image_path"].suffix.lower() if rec["image_path"] else ".jpg"

    for split_name, stems in splits.items():
        print(f"[INFO] split={split_name}, samples={len(stems)}", flush=True)
        split_t0 = time.time()

        for idx, stem in enumerate(stems, start=1):
            rec = index[stem]
            meta = rec["meta"]
            image_path = rec["image_path"]
            assert image_path is not None

            ext = image_path.suffix.lower()
            dst_image_path = out_dir / "images" / split_name / f"{stem}{ext}"
            dst_line_mask_path = out_dir / "line_mask" / split_name / f"{stem}.png"
            dst_center_path = out_dir / "center" / split_name / f"{stem}.png"
            dst_orient_path = out_dir / "orient" / split_name / f"{stem}.npy"

            t0 = time.time()
            convert_one_sample(
                meta=meta,
                image_path=image_path,
                dst_image_path=dst_image_path,
                dst_line_mask_path=dst_line_mask_path,
                dst_center_path=dst_center_path,
                dst_orient_path=dst_orient_path,
                cable_labels=args.labels,
                resize_to=resize_to,
                center_thickness=args.center_thickness,
                center_resize_threshold=args.center_resize_threshold,
                samples_per_px=args.samples_per_px,
                min_center_samples=args.min_center_samples,
                center_smooth_window=args.center_smooth_window,
                write_orient=not args.no_orient,
            )
            dt = time.time() - t0

            if idx % max(1, args.log_every) == 0:
                print(f"[{split_name}] {idx}/{len(stems)} {stem} done in {dt:.2f}s", flush=True)

        print(f"[INFO] split={split_name} finished in {time.time() - split_t0:.1f}s", flush=True)

    splits_dir = out_dir / "splits"
    for split_name, stems in splits.items():
        txt_name = "val.txt" if split_name == "val" else f"{split_name}.txt"
        save_split_txt(splits_dir / txt_name, stems, image_ext_map)

    print("[DONE] Conversion finished.", flush=True)
    print(f"Output root: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
