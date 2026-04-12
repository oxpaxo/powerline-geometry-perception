#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import importlib.util
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_module_from_path(py_path: Path):
    spec = importlib.util.spec_from_file_location("ttpla_converter_module", str(py_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def draw_polygon(ax, points, color="red", linewidth=1.2):
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] < 2:
        return
    closed = np.vstack([pts, pts[0]])
    ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=linewidth)


def collect_centerlines_from_shapes(mod, shapes, samples_per_px, min_center_samples, center_smooth_window):
    centerlines = []
    polygons = []

    for item in shapes:
        shape_type = str(item["shape_type"])
        points = item["points"]

        if shape_type == "polygon":
            polygons.append(points)
            center = mod.compute_polygon_centerline(
                polygon=points,
                samples_per_px=samples_per_px,
                min_samples=min_center_samples,
                smooth_window=center_smooth_window,
            )
            if center is not None and len(center) >= 2:
                centerlines.append(center)

        elif shape_type == "linestrip":
            center = mod.linestrip_to_centerline(
                points=points,
                samples_per_px=samples_per_px,
                min_samples=min_center_samples,
                smooth_window=max(3, center_smooth_window // 2 * 2 + 1),
            )
            if center is not None and len(center) >= 2:
                centerlines.append(center)

    return polygons, centerlines


def draw_centerlines(ax, centerlines, color="cyan", linewidth=2.0):
    for line in centerlines:
        pts = np.asarray(line, dtype=np.float32)
        if pts.shape[0] < 2:
            continue
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=linewidth)


def collect_raw_tangent_arrows(centerlines, arrow_stride=10, arrow_scale=40.0, negate_y=False):
    xs = []
    ys = []
    u = []
    v = []

    for line in centerlines:
        pts = np.asarray(line, dtype=np.float32)
        if pts.shape[0] < 2:
            continue

        for i in range(0, len(pts) - 1, arrow_stride):
            p0 = pts[i]
            p1 = pts[i + 1]

            dx = float(p1[0] - p0[0])
            dy = float(p1[1] - p0[1])

            norm = math.hypot(dx, dy)
            if norm <= 1e-6:
                continue

            vx = dx / norm
            vy = dy / norm

            mx = 0.5 * (p0[0] + p1[0])
            my = 0.5 * (p0[1] + p1[1])

            xs.append(mx)
            ys.append(my)
            u.append(vx * arrow_scale)
            v.append((-vy if negate_y else vy) * arrow_scale)

    return np.array(xs), np.array(ys), np.array(u), np.array(v)


def main():
    parser = argparse.ArgumentParser(description="Debug polygon -> centerline -> raw tangent")
    parser.add_argument("--converter", type=str, required=True, help="Path to convert_ttpla_to_v1.py")
    parser.add_argument("--json", type=str, required=True, help="Path to one LabelMe JSON")
    parser.add_argument("--image", type=str, default="", help="Optional image path override")
    parser.add_argument("--labels", nargs="+", default=["cable"])
    parser.add_argument("--samples-per-px", type=float, default=0.35)
    parser.add_argument("--min-center-samples", type=int, default=24)
    parser.add_argument("--center-smooth-window", type=int, default=5)
    parser.add_argument("--arrow-stride", type=int, default=10)
    parser.add_argument("--arrow-scale", type=float, default=40.0)
    parser.add_argument("--out", type=str, default="debug_polygon_centerline_tangent.png")
    args = parser.parse_args()

    converter_path = Path(args.converter)
    json_path = Path(args.json)
    out_path = Path(args.out)

    mod = load_module_from_path(converter_path)

    meta = mod.read_labelme_json(json_path)

    if args.image:
        image_path = Path(args.image)
    else:
        image_path = mod.infer_image_path(json_path, meta.get("imagePath", None))

    if image_path is None or not image_path.exists():
        raise FileNotFoundError(f"Cannot infer paired image for: {json_path}")

    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    shapes = mod.collect_cable_shapes(meta, args.labels)
    polygons, centerlines = collect_centerlines_from_shapes(
        mod=mod,
        shapes=shapes,
        samples_per_px=args.samples_per_px,
        min_center_samples=args.min_center_samples,
        center_smooth_window=args.center_smooth_window,
    )

    print("json:", json_path)
    print("image:", image_path)
    print("num_shapes:", len(shapes))
    print("num_polygons:", len(polygons))
    print("num_centerlines:", len(centerlines))

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # Panel 1: raw image
    axes[0].imshow(img)
    axes[0].set_title("raw image")
    axes[0].axis("off")

    # Panel 2: polygon + centerline
    axes[1].imshow(img)
    for poly in polygons:
        draw_polygon(axes[1], poly, color="red", linewidth=1.2)
    draw_centerlines(axes[1], centerlines, color="cyan", linewidth=2.0)
    axes[1].set_title("polygon + centerline")
    axes[1].axis("off")

    # Panel 3: raw tangent, image-coordinate display (v = dy)
    axes[2].imshow(img)
    draw_centerlines(axes[2], centerlines, color="cyan", linewidth=1.5)
    xs, ys, u, v = collect_raw_tangent_arrows(
        centerlines=centerlines,
        arrow_stride=args.arrow_stride,
        arrow_scale=args.arrow_scale,
        negate_y=False,
    )
    if len(xs) > 0:
        axes[2].quiver(
            xs, ys, u, v,
            color="yellow",
            angles="xy",
            scale_units="xy",
            scale=1,
            pivot="mid",
            width=0.003,
            headwidth=4,
            headlength=6,
            headaxislength=5,
        )
    axes[2].set_title("raw tangent (v = dy)")
    axes[2].axis("off")

    # Panel 4: raw tangent, negated-y display (v = -dy)
    axes[3].imshow(img)
    draw_centerlines(axes[3], centerlines, color="cyan", linewidth=1.5)
    xs2, ys2, u2, v2 = collect_raw_tangent_arrows(
        centerlines=centerlines,
        arrow_stride=args.arrow_stride,
        arrow_scale=args.arrow_scale,
        negate_y=True,
    )
    if len(xs2) > 0:
        axes[3].quiver(
            xs2, ys2, u2, v2,
            color="yellow",
            angles="xy",
            scale_units="xy",
            scale=1,
            pivot="mid",
            width=0.003,
            headwidth=4,
            headlength=6,
            headaxislength=5,
        )
    axes[3].set_title("raw tangent (v = -dy)")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=160, bbox_inches="tight")
    print("Saved to:", out_path)


if __name__ == "__main__":
    main()