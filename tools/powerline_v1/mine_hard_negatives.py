"""
Hard-negative mining script — placeholder.

STATUS: PLACEHOLDER — not implemented this round.

PLANNED USAGE:
    python tools/powerline_v1/mine_hard_negatives.py \
        --config configs/powerline_v1/powerline_v1_segformer_b1.py \
        --checkpoint work_dirs/.../iter_20000.pth \
        --data-root projects/powerline_v1/datasets/TTPLA \
        --split splits/train.txt \
        --out-file work_dirs/hard_negatives.json \
        --fp-threshold 0.5 \
        --dilate-radius 20

PLANNED PIPELINE:
    1. Load trained model + dataset
    2. For each image, run inference to get center_prob
    3. Load GT center mask, dilate by radius to get "near-wire zone"
    4. Find high-confidence predictions OUTSIDE near-wire zone = false positives
    5. Extract crop coordinates of FP regions
    6. Export as JSON: [{image, bbox, max_prob, area}, ...]
    7. Future: feed into sampler for hard-negative-aware training
"""

import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description='Mine hard-negative patches from model predictions')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--split', default='splits/train.txt')
    parser.add_argument('--out-file', default='work_dirs/hard_negatives.json')
    parser.add_argument('--fp-threshold', type=float, default=0.5,
                        help='Min center_prob to count as false positive')
    parser.add_argument('--dilate-radius', type=int, default=20,
                        help='Dilation radius around GT center to define near-wire zone')
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def main():
    args = parse_args()
    raise NotImplementedError(
        'Hard-negative mining is a placeholder this round. '
        'See docstring for planned pipeline.'
    )


if __name__ == '__main__':
    main()
