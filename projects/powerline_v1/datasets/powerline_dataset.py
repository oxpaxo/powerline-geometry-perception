import os.path as osp
from typing import List

from mmengine.fileio import list_dir_or_file, list_from_file
from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class PowerLineDataset(BaseSegDataset):
    METAINFO = dict(
        classes=('background', 'centerline'),
        palette=[[0, 0, 0], [255, 255, 255]],
    )

    def __init__(
        self,
        img_suffix='.jpg',
        seg_map_suffix='.png',
        orient_suffix='.npy',
        reduce_zero_label=False,
        **kwargs,
    ):
        self.orient_suffix = orient_suffix
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs,
        )

    def load_data_list(self) -> List[dict]:
        data_list = []

        img_dir = self.data_prefix.get('img_path', '')
        seg_dir = self.data_prefix.get('seg_map_path', '')
        orient_dir = self.data_prefix.get('orient_path', '')

        if self.ann_file is not None and osp.isfile(self.ann_file):
            lines = list_from_file(self.ann_file, backend_args=self.backend_args)
            basenames = [line.strip() for line in lines if line.strip()]
        else:
            img_names = list(
                list_dir_or_file(
                    dir_path=img_dir,
                    list_dir=False,
                    suffix=self.img_suffix,
                    recursive=False,
                    backend_args=self.backend_args,
                )
            )
            basenames = [osp.splitext(name)[0] for name in img_names]

        for basename in basenames:
            basename = basename.strip()
            basename = osp.splitext(basename)[0]

            img_path = osp.join(img_dir, basename + self.img_suffix)
            seg_map_path = osp.join(seg_dir, basename + self.seg_map_suffix)
            orient_path = osp.join(orient_dir, basename + self.orient_suffix)

            data_info = dict(
                img_path=img_path,
                seg_map_path=seg_map_path,
                orient_path=orient_path,
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label,
                seg_fields=[],
            )
            data_list.append(data_info)

        return data_list