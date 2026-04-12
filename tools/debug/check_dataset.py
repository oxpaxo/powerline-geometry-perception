from mmseg.utils import register_all_modules
register_all_modules(init_default_scope=True)

from mmengine.config import Config
from mmseg.registry import DATASETS

cfg = Config.fromfile('configs/powerline_v1/powerline_v1_r18_fpn.py')

dataset = DATASETS.build(cfg.train_dataloader['dataset'])
print('Dataset length =', len(dataset))

sample = dataset[0]

inputs = sample['inputs']
data_sample = sample['data_samples']

print('inputs.shape =', tuple(inputs.shape))
print('gt_sem_seg.shape =', tuple(data_sample.gt_sem_seg.data.shape))
print('gt_orient_map.shape =', tuple(data_sample.gt_orient_map.data.shape))
print('img_path =', data_sample.metainfo.get('img_path', None))
print('flip =', data_sample.metainfo.get('flip', None))
print('flip_direction =', data_sample.metainfo.get('flip_direction', None))
