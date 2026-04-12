from pathlib import Path
import sys
import os
import subprocess
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Run tools/test.py with multiple center thresholds')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/powerline_v1/powerline_v1_r18_fpn.py',
        help='Path to config file')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='work_dirs/powerline_v1_r18_fpn/iter_20000.pth',
        help='Path to checkpoint file')
    parser.add_argument(
        '--thresholds',
        type=float,
        nargs='+',
        default=[0.3, 0.4, 0.5, 0.6],
        help='Threshold list for model.test_cfg.center_threshold')
    parser.add_argument(
        '--out-root',
        type=str,
        default='work_dirs/powerline_v1_r18_fpn',
        help='Root directory to save threshold-specific test visualizations')
    return parser.parse_args()


def thr_to_tag(thr: float) -> str:
    # 0.3 -> thr03, 0.4 -> thr04, 0.55 -> thr055
    s = f'{thr:.2f}'.rstrip('0').rstrip('.')
    digits = s.replace('.', '')
    return f'thr{digits}'


def main():
    args = parse_args()

    root = Path(__file__).resolve().parents[2]
    os.chdir(root)

    config = Path(args.config)
    checkpoint = Path(args.checkpoint)
    out_root = Path(args.out_root)

    if not config.exists():
        raise FileNotFoundError(f'Config not found: {config}')
    if not checkpoint.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')

    print('Repo root =', root)
    print('Config =', config)
    print('Checkpoint =', checkpoint)
    print('Thresholds =', args.thresholds)
    print('Output root =', out_root)

    for thr in args.thresholds:
        tag = thr_to_tag(thr)
        show_dir = out_root / f'test_vis_{tag}'
        show_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            'tools/test.py',
            str(config),
            str(checkpoint),
            '--show-dir',
            str(show_dir),
            '--cfg-options',
            f'model.test_cfg.center_threshold={thr}',
        ]

        print('\n' + '=' * 80)
        print(f'Running threshold = {thr}')
        print('Command:')
        print(' '.join(cmd))
        print('=' * 80)

        subprocess.run(cmd, check=True)

    print('\nAll threshold runs finished.')


if __name__ == '__main__':
    main()